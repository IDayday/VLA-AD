from argparse import Namespace
from pathlib import Path

from scripts.last_vla_v2.two_expert_slot import prune_top3_eval_artifacts as prune


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_pruner_protects_navtest_top_checkpoint(tmp_path, monkeypatch):
    val_root = tmp_path / "val"
    nav_root = tmp_path / "nav"
    eliminated_file = tmp_path / "eliminated.txt"
    log_file = tmp_path / "prune.log"

    _write(
        val_root / "summary" / "val6000_summary.csv",
        "\n".join(
            [
                "split_name,checkpoint_name,checkpoint,PDMS",
                "val6000,a.ckpt,/tmp/a.ckpt,0.99",
                "val6000,b.ckpt,/tmp/b.ckpt,0.98",
                "val6000,c.ckpt,/tmp/c.ckpt,0.97",
                "val6000,epoch=142-step=115115.ckpt,/tmp/epoch=142-step=115115.ckpt,0.96",
                "",
            ]
        ),
    )
    _write(
        nav_root / "summary" / "navtest_summary.csv",
        "\n".join(
            [
                "split_name,checkpoint_name,checkpoint,PDMS",
                "navtest,epoch=142-step=115115.ckpt,/tmp/epoch=142-step=115115.ckpt,0.8728629594",
                "",
            ]
        ),
    )
    snapshot_dir = nav_root / "checkpoint_snapshots" / "epoch_142-step_115115.ckpt"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "epoch=142-step=115115.ckpt").write_text("ckpt", encoding="utf-8")

    monkeypatch.setattr(prune, "active_by_split", lambda args: (set(), set(), {}))

    args = Namespace(
        val_roots=[val_root],
        navtest_out_root=nav_root,
        ckpt_mirror_root=None,
        eliminated_file=eliminated_file,
        remote_hosts=["local"],
        top_k=3,
        protect_navtest_top_k=3,
        protect_names_file=None,
        poll_seconds=1.0,
        log_file=log_file,
        once=True,
    )

    prune.run_once(args)

    eliminated = eliminated_file.read_text(encoding="utf-8")
    assert "epoch=142-step=115115.ckpt" not in eliminated
    assert snapshot_dir.is_dir()
