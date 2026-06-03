from __future__ import annotations

from scripts.risk_vla.build_round1_evidence_report import build_evidence_report


def test_round1_evidence_report_contains_required_sections_and_terms(tmp_path):
    discovery = tmp_path / "discovery.md"
    discovery.write_text("# Inputs\n", encoding="utf-8")
    r0 = tmp_path / "r0.md"
    r0.write_text("# R0\n", encoding="utf-8")
    go = tmp_path / "go.md"
    go.write_text("Decision: `GO_R1_ONLY`\n", encoding="utf-8")
    output = tmp_path / "evidence.md"

    build_evidence_report(input_discovery_md=discovery, r0_diagnostics_md=r0, go_no_go_md=go, output_md=output)

    text = output.read_text(encoding="utf-8")
    for section in (
        "Current Branch / Commit",
        "Inputs Found And Missing",
        "R0 Risk-Head Diagnostic Results",
        "GO/NO-GO Decision",
        "R1 Oracle-Router Pilot Status",
        "R2 Predicted-Router Pilot Status",
    ):
        assert section in text
    assert "low-score risk scenarios" in text
    assert "path/terminal intent strategy" in text
    assert "not a final PDM performance claim" in text
