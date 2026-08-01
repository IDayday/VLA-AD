# Raw-input manifest

Raw experiment directories are read-only. Register evidence in
`result_sources.json`; do not copy or edit source results here. A source entry
has this form:

```json
{
  "id": "descriptive_source_id",
  "path": "repository/relative/result.csv",
  "enabled": true,
  "defaults": {
    "benchmark": "value verified from the evaluator/config",
    "split": "value verified from the evaluator/config",
    "method": "value verified from the run metadata"
  },
  "column_map": {
    "raw_scene_column": "scene_id",
    "raw_score_column": "aggregate_score"
  }
}
```

`defaults` only fill absent/null fields. They do not overwrite observed data.
Training/evaluation seeds must be present in the source or explicitly verified
before being added as defaults. The canonicalizer records source-relative paths,
row keys, hashes, mappings, and commands in `derived/`.
