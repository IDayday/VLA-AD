# 检查点审计

实际文件路径、字节数、SHA256 见 [checkpoint_identity.json](tables/checkpoint_identity.json)。重新读取文件并核验：

| 模型 | SHA256 |
|---|---|
| 官方 IL | `4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443` |
| 官方 RL | `23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20` |
| 共享 VLM safetensors | `79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c` |

RL 教师加载的是 active `action_head`。RL 检查点内的 old_policy/reference 参数全部明确列为忽略项，不进入教师。6 个缺失参数属于明确禁用的 action-aware 辅助路径，不参与当前前向；基础 DiT 必需参数缺失、unexpected 或 shape mismatch 均立即报错。

完整 missing_all、missing_core、unexpected、shape mismatch 和 ignored 名单在 [numerical_parity.json](tables/numerical_parity.json)。没有放宽核心层断言。E0-COPY 复制 RL 同结构模型后，逐步状态与最终轨迹 exact equality；这项验证已经实际运行，不等同于尚在运行的完整开发集 PDMS 评价。

旧 OPD 检查点只作为历史再现和局部诊断中的独立 roll-in 来源。它不冒充本轮大总体训练出的 warm start；主 P2/R2/C1 必须使用本轮按统一校准规则选出的各自训练种子的 warm checkpoint。
