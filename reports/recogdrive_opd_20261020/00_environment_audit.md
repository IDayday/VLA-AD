# 环境与数据审计

```json
{
  "owner": {
    "host": "training-vlawm-tsinghua-zt2",
    "pid": 1861373,
    "start_ticks": 1386428815,
    "code_sha": "0e789a8a876bd32f078724c4331983fc1ab9f996",
    "time": 1792425493.9735396
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
  "gpu": "NVIDIA A800-SXM4-80GB, GPU-cdd3d7ed-6094-0ef4-282c-2de8dbf86e00, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-c0a3d651-8691-b507-5df6-6c7f0cc6c897, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-b66ce64d-2b82-4ad3-f917-6ce384dfd9b6, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-220170c8-e900-2090-9c71-5eb562f2d0c0, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-ccacd0ea-138e-cba3-bdaa-c5290f852915, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-88994300-43aa-ec6a-973e-a92276b2a045, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-cb83ff92-fe81-0597-9d9a-257bfe48cec3, 81920 MiB\nNVIDIA A800-SXM4-80GB, GPU-c40bf31b-444d-5efb-7156-90b71a4d70b3, 81920 MiB\n",
  "data": {
    "splits": {
      "train": {
        "scenes": 4115,
        "logs": 50,
        "sha256": "e3292aae30586c02e6c0df62639a0dd3e9bfc58b55a6c5662ec4654e0bfaee1f"
      },
      "cal": {
        "scenes": 518,
        "logs": 4,
        "sha256": "10488343ac654799c109e80bf09d26da6b49e4fc4210e7e3dba53c13c87a82c3"
      },
      "dev": {
        "scenes": 1696,
        "logs": 16,
        "sha256": "f9552d196d8973a6cfd1f182702b777198012f9fe60d51acee08c039e75eea53"
      }
    },
    "source": "/mnt/project/VLA-AD/outputs/trajectory_effect_validation_20261019/manifests",
    "source_checkpoint": "frozen VLM encoding; the action projection is not cached",
    "caveat": "All splits are official navtrain; held out from this new training, potentially seen by original pretraining"
  },
  "initial_HEAD": "884ea1d0f67c60821d61033d61efb10916d16929"
}
```
