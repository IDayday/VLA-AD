# 原生动作规划器与DDIM审计

```json
{
  "parameters": 34329219,
  "flags": {
    "use_expert_features": false,
    "use_last_rd": false,
    "use_bit_drive": false,
    "use_risk_vla": false,
    "use_action_aware_aux": false,
    "sampling_method": "ddim",
    "num_inference_steps": 5,
    "input_embedding_dim": 384,
    "vlm_feature_dim": 1536,
    "action_horizon": 8,
    "action_dim": 3
  },
  "timesteps": [
    80,
    60,
    40,
    20,
    0
  ],
  "indices": [
    0,
    1,
    2,
    3,
    4
  ],
  "saved_sigma_buffer": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "eta_state": {
    "eta_logit": "tensor([inf], device='cuda:0')"
  },
  "sampler_note": "get_action deterministic: sigma=0; stochastic: actual EtaFixed then minimum1e-4, clipped standard-normal noise[-1,1]; final latent clip[-1,1]. sample_chain used by GRPO has a distinct minimum0.04. Gaussian KL rejected for clipped kernels."
}
```
