"""fim.py — Fill-In-The-Middle 数据增强

实现 FIM 论文 (Bavarian et al., 2022) 中的 PSM 和 SPM 两种变体。
FIM 变换在训练时对已 tokenize 的序列随机切割重排，使模型学会根据前后文预测中间内容。

PSM (Prefix-Suffix-Middle): <PRE> prefix <SUF> suffix <MID> middle
SPM (Suffix-Prefix-Middle): <PRE> <SUF> suffix <MID> prefix middle
"""
import functools
import numpy as np


@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    """从 SantaCoder tokenizer 的 additional_special_tokens 中提取 FIM token IDs。

    SantaCoder 的 additional_special_tokens 按固定顺序存储 5 个 token，
    第 2-5 个依次为 FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD。
    """
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
    """对 token 序列做 FIM 变换。

    以 fim_rate 概率触发变换；触发后以 fim_spm_rate 概率选择 SPM 或 PSM 格式。

    Args:
        sample: numpy array of token IDs
        np_rng: numpy RandomState
        fim_rate: 触发 FIM 变换的概率 (0.5 = 一半样本做 FIM)
        fim_spm_rate: FIM 变换中选择 SPM 格式的概率 (0.5 = SPM/PSM 各半)

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
