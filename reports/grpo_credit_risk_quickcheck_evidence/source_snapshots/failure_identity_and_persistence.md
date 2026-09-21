# Executed failure and subsequent repair (2026-09-21)

The original attempted-update initializer executed:

```python
p,m=load('a5_sft')             # this readonly loader froze every parameter
p.requires_grad_(True)        # BUG: destroys constructor-specific freeze mask
```

The archived constructor freezes EtaFixed and inactive planning-attention
branches. The incorrect initializer exposed 557 tensors, whereas preserving
constructor flags and applying the agent's exclusions yields 452 trainable
tensors. The 105 extras are fully listed in `audits/TINY_STEP1_COMPLETE.json`.

EtaFixed(base_eta=1) encodes eta_logit=atanh(1)=+inf, then uses tanh and `.item()`
to return a fixed finite eta. It is intentionally frozen in the archived source.
The failed displacement calculation included this tensor:

```python
current=state(p)
displacement=float(torch.sqrt(sum(
    (current[n].double()-start[n].double()).square().sum() for n in pars
)))
records.append(record)
save(dest/'steps.json',records)  # strict JSON rejects NaN from inf-inf
# Original ordering saved step1.pt only AFTER the JSON write: never reached.
```

All 8 attempts had completed one optimizer step before this logging exception.
No valid updated checkpoint was recovered. Full losses/tracebacks are preserved
in logs/tiny_*_step1.log and tiny/<seed>/<arm>/failure.log. This does not establish
that the updated model diverged; it establishes an implementation/identity
failure. No attempt was rerun and no step2–8 was executed.

Repair: preserve the actual constructor's requires_grad flags, explicitly
validate the trainable mask and finite trainable values before optimizer use,
write post-update checkpoint before diagnostics, and refuse any stopped-arm
restart. The repair passed read-only identity checks and regression tests.
This file documents the executed snippets and forensic reconstruction; it is
not represented as a contemporaneous full-source snapshot of the failed jobs.
