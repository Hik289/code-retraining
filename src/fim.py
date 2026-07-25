"""Fill-in-the-middle augmentation for all supported model families.

Implements the PSM and SPM variants from Bavarian et al. (2022). FIM tokens
are supplied by the model configuration.
"""
import numpy as np


def get_fim_token_ids(tokenizer, config):
    """Resolve FIM token IDs from a model configuration.

    Args:
        tokenizer: HuggingFace tokenizer
        config: dict from load_model_config(), must contain fim_prefix/middle/suffix

    Returns:
        (suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id)
        pad_tok_id may be None if the model has no FIM pad token (e.g. Code Llama).
    """
    prefix_id = tokenizer.convert_tokens_to_ids(config["fim_prefix"])
    middle_id = tokenizer.convert_tokens_to_ids(config["fim_middle"])
    suffix_id = tokenizer.convert_tokens_to_ids(config["fim_suffix"])

    pad_token = config.get("fim_pad")
    if pad_token is not None:
        pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    else:
        pad_id = None

    # An unknown-token result means the configured FIM token is not in the vocabulary.
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
    """Apply an optional FIM permutation to a token sequence.

    The transform is applied with probability ``fim_rate``. Conditional on a
    transform, ``fim_spm_rate`` selects SPM instead of PSM.

    PSM (Prefix-Suffix-Middle): <PRE> prefix <SUF> suffix <MID> middle
    SPM (Suffix-Prefix-Middle): <PRE> <SUF> suffix <MID> prefix middle

    Args:
        sample: numpy array of token IDs
        np_rng: numpy RandomState
        suffix_tok_id, prefix_tok_id, middle_tok_id: FIM token IDs
        pad_tok_id: FIM pad token ID (unused in current impl, reserved)
        fim_rate: Probability of applying a FIM transform.
        fim_spm_rate: Conditional probability of selecting SPM.

    Returns:
        (new_sample, np_rng)
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = sample[: boundaries[0]]
        middle = sample[boundaries[0] : boundaries[1]]
        suffix = sample[boundaries[1] :]

        if np_rng.binomial(1, fim_spm_rate):
            # SPM: <PRE> <SUF> suffix <MID> prefix middle
            new_sample = np.concatenate(
                [[prefix_tok_id, suffix_tok_id], suffix,
                 [middle_tok_id], prefix, middle]
            )
        else:
            # PSM: <PRE> prefix <SUF> suffix <MID> middle
            new_sample = np.concatenate(
                [[prefix_tok_id], prefix, [suffix_tok_id],
                 suffix, [middle_tok_id], middle]
            )
        sample = new_sample

    return sample, np_rng
