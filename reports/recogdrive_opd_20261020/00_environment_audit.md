# 环境与数据审计

```json
{
  "owner": {
    "host": "training-rl-zt2",
    "pid": 1681556,
    "start_ticks": 1379275463,
    "code_sha": "3fa2fc68515be7f01190e45f35834b7c9b0d7d9d",
    "time": 1792435070.5730386
  },
  "python": "3.9.25 (main, Nov  3 2025, 22:33:05) \n[GCC 11.2.0]",
  "versions": {
    "torch": "2.5.1+cu124",
    "transformers": "4.57.6",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "pytest": "8.4.2",
    "safetensors": "0.7.0"
  },
  "gpu": "NVIDIA A800-SXM4-80GB, GPU-f9371c55-5054-e42c-71bf-f1314992af79, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-42e1151c-5d53-6fcc-39a7-6ca10eec54f6, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-72a4835c-963d-0534-e867-0fbdbb2cbdf2, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-659076b8-990e-4ef1-2a38-c2aea273ab0c, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-89896b99-8541-74ca-2e23-cd59843a102a, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-7b4dce95-2581-1a46-2a50-7a1ac5498b5f, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-23c150c4-0b05-986a-bf55-cdd3f4fec999, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-1d07169b-659c-d3b9-43c5-c897e137ab6e, 81920 MiB\n",
  "data": {
    "splits": {
      "train": {
        "scenes": 4115,
        "logs": 50,
        "sha256": "3630e5eb39b8de72300edbe5544611df6d5991473e81c0249306ae7d763c8e55"
      },
      "cal": {
        "scenes": 518,
        "logs": 4,
        "sha256": "158a70d820ca8d0eea8138eaef9f69556616f746bab077392fb683acd0dbc7e5"
      },
      "dev": {
        "scenes": 1696,
        "logs": 16,
        "sha256": "7793321e859e13c6e5740031b69869e4954fb99c3a5b81d62ee1384224eafeb3"
      }
    },
    "source": "/mnt/project/VLA-AD/outputs/trajectory_effect_validation_20261019/manifests",
    "source_checkpoint": "frozen VLM encoding; the action projection is not cached",
    "caveat": "All splits are official navtrain; held out from this new training, potentially seen by original pretraining"
  },
  "initial_HEAD": "884ea1d0f67c60821d61033d61efb10916d16929"
}
```

## 最终资源与数据身份

真实核验40张互异A800，主机为training-rl-zt2, training-vla-zt, training-vlawm-tsinghua-zt2, training-vlawm-tsinghua-zt3, training-vlawm-tsinghua-zt5。逐主机CUDA/驱动/PyTorch环境与GPU UUID见tables/audit_hardware_hosts.json。原工作区未提交修改保留，研究在独立worktree中进行，原有ReCogDrive默认源码未改动。

版本化索引位于configs/opd/splits，含训练4115、校准518、开发1696、官方测试12146个精确有序场景；206个源log的字节哈希已核验。driver_record保留车辆标识，另以更保守的capture timestamp检查跨用途重叠，均无交集。真实新目录重建已逐字段、逐顺序核验，不依赖旧研究输出manifest。原始数据、图像、GT及大缓存均不进git。
