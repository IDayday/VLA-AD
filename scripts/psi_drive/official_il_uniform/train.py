"""Launch the official IL recipe with uniform single-trajectory PSI supervision."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]
STAGES = {"single": (1, 16, "1e-4", "1e-6"), "dual": (2, 8, "1e-4", "1e-6"),
          "batch256": (2, 16, "2e-4", "2e-6")}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    nodes, batch, lr, min_lr = STAGES[args.stage]
    if nodes == 2 and (not args.master_addr or not args.resume):
        parser.error("two-node stages require --master-addr and --resume")
    if not 0 <= args.node_rank < nodes:
        parser.error("--node-rank is outside the current stage")
    launcher = [sys.executable, "-m", "torch.distributed.run", f"--nnodes={nodes}", "--nproc_per_node=8"]
    if nodes == 1:
        launcher += ["--standalone"]
    else:
        launcher += [f"--node_rank={args.node_rank}", f"--master_addr={args.master_addr}",
                     f"--master_port={args.master_port}", "--rdzv_conf=timeout=150"]
    output = json.dumps(str(args.output.resolve()))
    overrides = {
        "agent": "recogdrive_agent", "agent.lr": lr, "agent.grpo": "false",
        "agent.vlm_path": json.dumps(str(args.stage1.resolve())),
        "agent.cam_type": "single", "agent.cache_hidden_state": "true", "agent.vlm_type": "internvl",
        "agent.dit_type": "small", "agent.vlm_size": "small", "agent.sampling_method": "ddim",
        "agent.use_expert_features": "false", "agent.use_jepa": "false", "agent.use_vggt": "false",
        "agent.checkpoint_path": "", "agent.allow_random_init": "true",
        "agent.train_backbone": "false", "agent.stage3_objective": "none",
        "agent.use_last_vla": "false", "agent.use_last_rd": "false",
        "agent.scheduler_epochs": 200, "agent.scheduler_warmup_epochs": 3, "agent.scheduler_min_lr": min_lr,
        "trainer.params.max_epochs": 200, "trainer.params.devices": 8,
        "trainer.params.accumulate_grad_batches": 1, "trainer.params.default_root_dir": output,
        "trainer.params.num_nodes": nodes,
        "dataloader.params.batch_size": batch, "dataloader.params.num_workers": 8,
        "dataloader.params.prefetch_factor": 2,
        "train_test_split": "navtrain", "cache_path": json.dumps(str(args.cache.resolve())),
        "use_cache_without_dataset": "true", "force_cache_computation": "false",
        "seed": 0, "output_dir": output, "experiment_name": "official_il_psi_uniform",
        "cache_train_all_records": "false", "stage2_target_source": "pareto_support",
        "stage2_pareto_support_index_path": json.dumps(str(args.support.resolve())),
        "stage2_pareto_require_index": "true", "stage2_pareto_log_diagnostics": "true",
        "hydra.run.dir": "${output_dir}/hydra_rank_${oc.env:RANK,0}", "hydra.output_subdir": "code/hydra",
    }
    if args.resume:
        overrides["+resume_ckpt_path"] = json.dumps(str(args.resume.resolve()))
    command = launcher + ["navsim/planning/script/run_training_recogdrive.py"]
    command.extend(f"{key}={value}" for key, value in overrides.items())
    env = dict(os.environ, RECOGDRIVE_KEY_STEPS="50000,60000,80000,100000,120000,140000,160000")
    if args.render:
        print(shlex.join(command))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)

if __name__ == "__main__":
    main()
