#!/usr/bin/env python3
"""Verify archive bytes, source closure and actual session splits, without imports."""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(base, source_override=None):
    manifest = json.loads((base / "SOURCE_MANIFEST.json").read_text())
    source = source_override or base / "ReCogDrive"
    indexed = {v["path"]: v for v in manifest["files"]}
    bad = []
    for rel, rec in indexed.items():
        p = source / rel
        if not p.is_file() or p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]:
            bad.append(rel)
    assert not bad, {"byte_mismatch": bad}
    syntax_bad = []
    imports = {}
    modules = defaultdict(list)
    for rel in indexed:
        if not rel.endswith(".py"):
            continue
        modules[Path(rel).stem].append(rel)
        try:
            t = ast.parse((source / rel).read_text(), filename=rel)
            deps = set()
            for n in ast.walk(t):
                if isinstance(n, ast.Import):
                    deps.update(x.name for x in n.names)
                elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                    deps.add(n.module)
            imports[rel] = deps
        except (SyntaxError, UnicodeError) as e:
            syntax_bad.append({"path": rel, "error": str(e)})
    assert not syntax_bad, {"python_syntax_errors": syntax_bad}
    def local_modules(name):
        if "." not in name:
            return [p for p in modules.get(name, []) if "/reference/original_sources/" not in p]
        suffixes = ("/" + name.replace(".", "/") + ".py", "/" + name.replace(".", "/") + "/__init__.py")
        return [p for p in indexed if p.endswith(suffixes)]
    # Frozen reference sources retain their original bytes. Their parents are
    # archival locations, so execute the complete research tree, not these copies.
    prefix = "1026-interface_teacher_portable/repo/reference/original_sources/"
    original = sorted(k for k in indexed if k.startswith(prefix) and k.endswith(".py"))
    assert len(original) == 8, len(original)
    comparisons = []
    queue = []
    for rel in original:
        candidates = [x for x in modules[Path(rel).stem] if "/reference/original_sources/" not in x]
        matching = [x for x in candidates if indexed[x]["sha256"] == indexed[rel]["sha256"]]
        assert matching, {"original_source_revision_missing": rel, "candidates": candidates}
        comparisons.append(dict(frozen=rel, byte_identical_research_sources=matching))
        queue.extend(matching)
    seen = set()
    external = set()
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for name in imports.get(rel, set()):
            local = local_modules(name)
            if local:
                queue.extend(local)
            elif name.split(".")[0] not in sys.stdlib_module_names:
                external.add(name.split(".")[0])
    expected = {"ol_l1_data_pack_v1", "ol_stage_b_final_factory_v1", "ol_l3_outer_refit_v1",
                "p2_u1_paired_assets_v1", "p2_u1_paired_mapping_math_v1", "r2_counterfactual", "g5_runtime"}
    assert expected <= set(modules), sorted(expected - set(modules))
    # Explicitly separate Python libraries from missing research modules.
    allowed_external = {"numpy", "scipy", "torch", "torchvision", "transformers", "diffusers",
                        "PIL", "pandas", "sklearn", "timm", "einops", "omegaconf", "hydra",
                        "nuplan", "navsim", "model", "tqdm", "matplotlib", "plotly", "cv2",
                        "shapely", "pyquaternion", "imageio", "wandb", "datasets", "safetensors",
                        "huggingface_hub", "pytorch_lightning", "lightning", "flash_attn", "yaml"}
    unresolved = sorted(n for n in external if n.startswith(("ol_", "p2_", "b5_", "r2_", "r3_", "r4_", "g5_", "full_six_")))
    assert not unresolved, {"unresolved_research_or_unknown_external_modules": unresolved}

    ol = source / "1024-representation_distillation_diagnosis_batch5/stage2/offline_library"
    split = json.loads((ol / "preparation/session_cv_metadata_v1/SPLITS.json").read_text())
    selected = {r["index"] for r in split["selected"]}
    assert len(selected) == len(split["selected"]) == 10042
    assert len({r["session"] for r in split["selected"]}) == 130
    assert not split["train_validation_session_intersection"]
    by_index = {r["index"]: r for r in split["selected"]}

    def partition(a, b, population):
        ai, bi = set(a["selected_indices"]), set(b["selected_indices"])
        sa, sb = set(a["sessions"]), set(b["sessions"])
        assert len(ai) == len(a["selected_indices"]) and len(bi) == len(b["selected_indices"])
        assert not ai & bi and ai | bi == population
        assert not sa & sb
        for z in (a, b):
            assert {by_index[i]["session"] for i in z["selected_indices"]} == set(z["sessions"])
        return dict(a_samples=len(ai), b_samples=len(bi), a_sessions=len(sa), b_sessions=len(sb), disjoint=True)

    folds = []
    held_indices = set()
    held_sessions = set()
    for f in split["folds"]:
        outer = partition(f["outer_train"], f["outer_held"], selected)
        inner = partition(f["inner_fit"], f["inner_calibration"], set(f["outer_train"]["selected_indices"]))
        assert not held_indices & set(f["outer_held"]["selected_indices"])
        assert not held_sessions & set(f["outer_held"]["sessions"])
        held_indices.update(f["outer_held"]["selected_indices"])
        held_sessions.update(f["outer_held"]["sessions"])
        folds.append(dict(fold=f["fold"], outer=outer, inner=inner))
    assert held_indices == selected and len(held_sessions) == 130 and len(folds) == 5
    final = partition(split["final_locked_family_inner_fit"], split["final_locked_family_inner_calibration"], selected)
    assert (final["a_samples"], final["b_samples"], final["a_sessions"], final["b_sessions"]) == (8021, 2021, 104, 26)
    return dict(status="PASS", files_verified=len(indexed), bytes_verified=sum(v["bytes"] for v in indexed.values()),
                python_files_checked=len(imports), syntax_errors=syntax_bad,
                original_eight_sources=comparisons, critical_local_source_closure=sorted(seen),
                required_modules_present=sorted(expected), critical_external_python_packages=sorted(external),
                additional_external_packages_for_review=sorted(external - allowed_external),
                missing_critical_research_modules=unresolved,
                split=dict(seed=split["seed"], selected=10042, sessions=130, final=final, folds=folds),
                model_imports=0, model_forwards=0, training_calls=0,
                scope="Static dependency/source-byte and real split validation; not an inference/training reproduction.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    print(json.dumps(verify(args.root.resolve()), ensure_ascii=False, indent=2))
