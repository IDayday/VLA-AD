# 严格权重加载审计

```json
{
  "IL": {
    "path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt",
    "sha256": "4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443",
    "load_report": {
      "path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt",
      "missing": [],
      "missing_all": [
        "action_aware_aux_head.0.bias",
        "action_aware_aux_head.0.weight",
        "action_aware_aux_head.1.bias",
        "action_aware_aux_head.1.weight",
        "action_aware_aux_head.3.bias",
        "action_aware_aux_head.3.weight"
      ],
      "missing_noncore": [
        "action_aware_aux_head.0.bias",
        "action_aware_aux_head.0.weight",
        "action_aware_aux_head.1.bias",
        "action_aware_aux_head.1.weight",
        "action_aware_aux_head.3.bias",
        "action_aware_aux_head.3.weight"
      ],
      "missing_semantics": "missing lists required core tensors; missing_all includes explicitly disabled auxiliary targets left unrestored",
      "unexpected": [],
      "shape_mismatch": [],
      "ignored": [],
      "restored_tensors": 347,
      "dtypes": [
        "torch.float32",
        "torch.int64"
      ]
    }
  },
  "RL": {
    "path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt",
    "sha256": "23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20",
    "load_report": {
      "path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt",
      "missing": [],
      "missing_all": [
        "action_aware_aux_head.0.bias",
        "action_aware_aux_head.0.weight",
        "action_aware_aux_head.1.bias",
        "action_aware_aux_head.1.weight",
        "action_aware_aux_head.3.bias",
        "action_aware_aux_head.3.weight"
      ],
      "missing_noncore": [
        "action_aware_aux_head.0.bias",
        "action_aware_aux_head.0.weight",
        "action_aware_aux_head.1.bias",
        "action_aware_aux_head.1.weight",
        "action_aware_aux_head.3.bias",
        "action_aware_aux_head.3.weight"
      ],
      "missing_semantics": "missing lists required core tensors; missing_all includes explicitly disabled auxiliary targets left unrestored",
      "unexpected": [],
      "shape_mismatch": [],
      "ignored": [
        {
          "key": "agent.action_head.old_policy.ddpm_betas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_alphas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_alphas_cumprod_prev",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_one_minus_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_recip_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_recipm1_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_var",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_logvar_clipped",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_mu_coef1",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_mu_coef2",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_t_schedule",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_t",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_alphas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_alphas_sqrt",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_alphas_prev",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_sqrt_one_minus_alphas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_sigmas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.rotary_embedder.inv_freq",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.norm_final.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.linear.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.linear.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.modulation_proj.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.modulation_proj.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.norm.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.norm.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.feature_encoder.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.feature_encoder.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.fusion_projector.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.fusion_projector.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.norm.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.position_embedding.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.eta.eta_logit",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        }
      ],
      "restored_tensors": 347,
      "dtypes": [
        "torch.float32",
        "torch.int64"
      ]
    }
  },
  "COPY": {
    "path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt",
    "sha256": "23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20",
    "load_report": {
      "path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt",
      "missing": [],
      "missing_all": [
        "action_aware_aux_head.0.bias",
        "action_aware_aux_head.0.weight",
        "action_aware_aux_head.1.bias",
        "action_aware_aux_head.1.weight",
        "action_aware_aux_head.3.bias",
        "action_aware_aux_head.3.weight"
      ],
      "missing_noncore": [
        "action_aware_aux_head.0.bias",
        "action_aware_aux_head.0.weight",
        "action_aware_aux_head.1.bias",
        "action_aware_aux_head.1.weight",
        "action_aware_aux_head.3.bias",
        "action_aware_aux_head.3.weight"
      ],
      "missing_semantics": "missing lists required core tensors; missing_all includes explicitly disabled auxiliary targets left unrestored",
      "unexpected": [],
      "shape_mismatch": [],
      "ignored": [
        {
          "key": "agent.action_head.old_policy.ddpm_betas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_alphas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_alphas_cumprod_prev",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_one_minus_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_recip_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_sqrt_recipm1_alphas_cumprod",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_var",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_logvar_clipped",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_mu_coef1",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddpm_mu_coef2",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_t_schedule",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_t",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_alphas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_alphas_sqrt",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_alphas_prev",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_sqrt_one_minus_alphas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ddim_sigmas",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.timestep_encoder.timestep_embedder.linear_2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.rotary_embedder.inv_freq",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.0.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.1.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.2.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.3.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.4.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.5.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.6.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.7.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.8.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.9.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.10.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.11.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.12.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.13.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.14.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.norm1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_q.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_q.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_k.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_k.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_v.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_v.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.q_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.k_norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_out.0.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.attn.to_out.0.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.norm2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w12.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w12.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.ffn.w3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.adaLN_modulation.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.transformer_blocks.15.adaLN_modulation.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.norm_final.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.linear.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.linear.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.modulation_proj.1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.model.final_layer.modulation_proj.1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.norm.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.his_traj_encoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.norm.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.ego_status_encoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc3.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_encoder.fc3.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.feature_encoder.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.feature_encoder.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.fusion_projector.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.fusion_projector.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc1.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc1.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.norm.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.norm.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc2.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.action_decoder.fc2.bias",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.position_embedding.weight",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        },
        {
          "key": "agent.action_head.old_policy.eta.eta_logit",
          "reason": "inactive RL/IL reference, never selected as active teacher"
        }
      ],
      "restored_tensors": 347,
      "dtypes": [
        "torch.float32",
        "torch.int64"
      ]
    }
  },
  "VLM": {
    "path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B",
    "sha256": "79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c",
    "deployment_shared": true,
    "limitation": "IL/RL action-head checkpoints omit VLM tensors; original training VLM cannot be proved from these files alone"
  }
}
```

## 官方模型来源

本地三个权重文件的实际SHA256均与公开发布文件的LFS SHA256完全相符；固定发布revision如下：

- IL：[固定发布文件](https://huggingface.co/owl10/ReCogDrive-2B-IL/blob/0fa96f2159d5bc0f816ecd3818f00541db9db6ff/ReCogDrive_Diffusion_Planner_2B_IL.ckpt)，revision `0fa96f2159d5bc0f816ecd3818f00541db9db6ff`，SHA256 `4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443`。
- RL：[固定发布文件](https://huggingface.co/owl10/ReCogDrive-2B-RL/blob/7346ede73f03ed577811024f1a5d16fb063a47b9/ReCogDrive_Diffusion_Planner_2B_RL.ckpt)，revision `7346ede73f03ed577811024f1a5d16fb063a47b9`，SHA256 `23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20`。
- VLM：[固定发布文件](https://huggingface.co/owl10/ReCogDrive-VLM-2B/blob/16873acca08e3c04ab229b3d973f39aeba9db68d/model.safetensors)，revision `16873acca08e3c04ab229b3d973f39aeba9db68d`，SHA256 `79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c`。

IL/RL共有347个活动张量，326个活动张量不同；RL old_policy与IL逐张量完全相同。教师只加载活动头。历史VLM训练身份不能由仅动作头checkpoint证明，本研究显式采用同一冻结官方VLM。
