import functools
import numpy as np


@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        _, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = (
            tokenizer.special_tokens_map["additional_special_tokens"]
        )
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok]
            for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            None, None, None, None
        )
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


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
