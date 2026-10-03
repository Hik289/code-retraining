import numpy as np


def get_fim_token_ids(tokenizer, config):
    prefix_id = tokenizer.convert_tokens_to_ids(config["fim_prefix"])
    middle_id = tokenizer.convert_tokens_to_ids(config["fim_middle"])
    suffix_id = tokenizer.convert_tokens_to_ids(config["fim_suffix"])

    pad_token = config.get("fim_pad")
    if pad_token is not None:
        pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    else:
        pad_id = None

    unk_id = getattr(tokenizer, "unk_token_id", None)
    for name, tid in [("prefix", prefix_id), ("middle", middle_id), ("suffix", suffix_id)]:
        if tid == unk_id:
            raise ValueError(
                f"FIM {name} token '{config[f'fim_{name}']}' resolved to unk_token_id={unk_id} "
                f"for model {config['short_name']}. Check config."
            )

    return suffix_id, prefix_id, middle_id, pad_id


def permute(sample, np_rng, suffix_tok_id, prefix_tok_id, middle_tok_id,
            pad_tok_id, fim_rate=0.5, fim_spm_rate=0.5):
    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = sample[: boundaries[0]]
        middle = sample[boundaries[0] : boundaries[1]]
        suffix = sample[boundaries[1] :]

        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [[prefix_tok_id, suffix_tok_id], suffix,
                 [middle_tok_id], prefix, middle]
            )
        else:
            new_sample = np.concatenate(
                [[prefix_tok_id], prefix, [suffix_tok_id],
                 suffix, [middle_tok_id], middle]
            )
        sample = new_sample

    return sample, np_rng
