# Main-text suggestions (not applied automatically)

These edits preserve the paper's architecture and conclusions.  They address
only nomenclature, arithmetic interpretation, provenance, and claims whose
current wording is stronger than the available evidence.

1. **Normalize the Stage-3 name.**  On physical PDF page 2, replace
   “Iterative Pareto-guided policy refinement (IPR)” with **“Adaptive
   Pareto-Guided Policy Refinement (APR)”**.  The rest of the paper and this
   supplement use APR.

2. **Use percentage points for hard-scene recovery.**  Replace the abstract's
   “11.1% higher” with **“11.1 percentage points higher”**.  The displayed
   counts give `440/658 - 367/658 = 11.09` percentage points; this is not an
   11.1% relative change.

3. **Qualify the 658-scene checkpoint identity.**  State that the 367/440
   comparison uses the historical paper recovery checkpoint.  The paired
   transition is 93 repairs, 20 new failures, 347 retained successes, and 198
   persistent failures, so net repair is 73.  Do not attach this matrix to the
   later 91.45 full-benchmark checkpoint; its hard-set result is different and
   remains audit-only.

4. **Explain the GT-only 86.4/86.5 difference.**  Table 2 reports 86.4 while
   Table 4 reports 86.5.  If these are different checkpoints or protocols, add
   a footnote; if the difference is only rounding, normalize the displayed
   precision.  No source was found that justifies silently merging the rows.

5. **Bound the APR causal wording.**  Table 5 establishes a monotonic
   single-lineage sequence (91.10, 91.21, 91.37, 91.45), but by itself does not
   isolate dynamic teacher reconstruction from extra optimization steps.  Use
   wording such as “consistent with dynamic refinement” until the step-matched
   fixed-teacher/extra-training controls are run.

6. **Align the PC-MTS compatibility statement with the released code.**  The
   paper describes average distance to the K nearest rollouts; the audited
   historical implementation selects the single nearest rollout.  Either
   release the KNN implementation/run manifest or describe the archived branch
   as the K=1 realization and make K=3 a proposed reproduction setting.

7. **Resolve the FF-PGRPO epoch manifest.**  The main paper reports 10 epochs;
   one archived Core-Pareto launcher records 20.  Preserve 10 as the paper
   protocol, label the 20-epoch artifact historical, and archive the resolved
   config for the exact paper checkpoint.

8. **Clarify evaluation/model selection.**  The final NAVSIM v1 row is
   scene-reproduced on 12,138 scored `navtest` scenes, but `navtest` was also
   used in checkpoint selection in the archived workflow.  Avoid describing it
   as a clean held-out model-selection estimate.  State one-trajectory
   inference and no test-time scorer/reranking consistently in all result
   captions.

9. **Distinguish reported baselines from local reproduction.**  Only AMPT's
   final v1/v2 rows have local scene-level exports in the audited tree.  Mark
   other headline baselines as reported unless their official artifacts and
   identical scorer/data/sensor protocol are supplied; avoid “SOTA” unless the
   comparison is demonstrably protocol-matched.

10. **Add appendix references.**  Point the method section to Appendix A/B for
    metric partitions, coherent references, and the three algorithms; point
    the experimental section to Appendix C--G for provenance, the reported
    supervision reversal, APR diagnostics, and the fixed 658-scene transition.

11. **Resolve the camera-input manifest.**  The paper describes multi-view
    input, while one audited APR resolved command records `cam_type=single`.
    Archive the exact final-checkpoint command and either align the architecture
    text or explain which stages/results use each camera setting.
