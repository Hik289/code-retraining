import json
import subprocess
import uuid


RUNNER = """
import json
import os
import signal
import sys

payload = json.load(sys.stdin)
exit_process = os._exit
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
scope = {'__name__': '__main__'}
signal.signal(signal.SIGALRM, lambda *_: exit_process(29))
signal.setitimer(signal.ITIMER_REAL, payload['timeout'])
try:
    exec(compile(payload['content'], '<candidate>', 'exec'), scope)
    if payload.get('test_code'):
        exec(compile(payload['test_code'], '<tests>', 'exec'), scope)
except ModuleNotFoundError:
    exit_process(21)
except ImportError:
    exit_process(20)
except (SyntaxError, IndentationError):
    exit_process(22)
except NameError:
    exit_process(23)
except AttributeError:
    exit_process(24)
except TypeError:
    exit_process(25)
except ValueError:
    exit_process(26)
except AssertionError:
    exit_process(28)
except BaseException:
    exit_process(27)
exit_process(0)
"""

STATUSES = {
    0: "ok", 20: "ImportError", 21: "ModuleNotFoundError",
    22: "SyntaxError", 23: "NameError", 24: "AttributeError",
    25: "TypeError", 26: "ValueError", 27: "other_error",
    28: "AssertionError", 29: "timeout", 137: "resource_limit",
}


class ExecutionVerifier:
    def __init__(self, image="python:3.11-slim", timeout=5):
        if timeout <= 0:
            raise ValueError("Execution timeout must be positive")
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=True, capture_output=True, text=True, timeout=30,
        )
        self.image = result.stdout.strip()
        if not self.image.startswith("sha256:"):
            raise RuntimeError("Cannot resolve the local execution image")
        self.timeout = timeout

    def __call__(self, sample):
        name = "retraining-" + uuid.uuid4().hex
        command = [
            "docker", "run", "--rm", "--pull", "never", "--name", name,
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "64",
            "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
            "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "-i", self.image,
            "python", "-I", "-c", RUNNER,
        ]
        try:
            result = subprocess.run(
                command, input=json.dumps(dict(sample, timeout=self.timeout)), text=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=self.timeout + 30,
            )
            if result.returncode in (125, 126, 127):
                raise RuntimeError(f"Execution container failed: {result.stderr.strip()}")
            return STATUSES.get(result.returncode, "other_error")
        except subprocess.TimeoutExpired:
            return "container_timeout"
        finally:
            cleanup = subprocess.run(
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, timeout=15,
            )
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                raise RuntimeError(f"Execution cleanup failed: {cleanup.stderr.strip()}")
