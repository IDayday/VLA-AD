# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .internvit_liger_monkey_patch import apply_liger_kernel_to_internvit
from .llama_rmsnorm_monkey_patch import \
    replace_llama_rmsnorm_with_fused_rmsnorm
from .pad_data_collator import (concat_pad_data_collator,
                                dpo_concat_pad_data_collator,
                                pad_data_collator)
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler


# Packed-training and legacy FlashAttention patches depend on private
# Transformers symbols that vary by release. Import them only when the
# corresponding feature is explicitly enabled; ordinary SFT does not use them.
def replace_internlm2_attention_class(*args, **kwargs):
    from .internlm2_packed_training_patch import replace_internlm2_attention_class as implementation
    return implementation(*args, **kwargs)


def replace_qwen2_attention_class(*args, **kwargs):
    from .qwen2_packed_training_patch import replace_qwen2_attention_class as implementation
    return implementation(*args, **kwargs)


def replace_phi3_attention_class(*args, **kwargs):
    from .phi3_packed_training_patch import replace_phi3_attention_class as implementation
    return implementation(*args, **kwargs)


def replace_llama_attention_class(*args, **kwargs):
    from .llama_packed_training_patch import replace_llama_attention_class as implementation
    return implementation(*args, **kwargs)


def replace_llama2_attn_with_flash_attn(*args, **kwargs):
    from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn as implementation
    return implementation(*args, **kwargs)


def replace_llama_attn_with_flash_attn(*args, **kwargs):
    from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn as implementation
    return implementation(*args, **kwargs)

__all__ = ['replace_llama_attn_with_flash_attn',
           'replace_llama_rmsnorm_with_fused_rmsnorm',
           'replace_llama2_attn_with_flash_attn',
           'replace_train_sampler',
           'replace_train_dataloader',
           'replace_internlm2_attention_class',
           'replace_qwen2_attention_class',
           'replace_phi3_attention_class',
           'replace_llama_attention_class',
           'pad_data_collator',
           'dpo_concat_pad_data_collator',
           'concat_pad_data_collator',
           'apply_liger_kernel_to_internvit']
