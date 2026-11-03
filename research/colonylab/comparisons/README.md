# Team source comparison

These fixtures are imported with ColonyLab under `research/colonylab/` in the
combined public repository. They preserve the historical comparison and do not
replace the repository-root `p113` package or standalone commands.

`qam_p113_baseline/` contains the exact eight-file source subset used by the
controlled comparison, taken from
[`smar-01/qam-p113-latency` at `62d6dcd`](https://github.com/smar-01/qam-p113-latency/tree/62d6dcd8af9ecd7136096dbcce3eb40371987422).
Its manifest binds the canonical LF Git source bytes for every file. The
historical local audit used CRLF checkout bytes; executable source text matches
after newline normalization. This is a historical comparison fixture,
not a complete runnable checkout. Upstream attribution is retained; no new
license is asserted for upstream material.

`qam_p113_corrected/` contains the local corrected implementation. Its default
quality mode explicitly uses the enclosing ColonyLab package. The
[standalone upstream fix proposal](https://github.com/smar-01/qam-p113-latency/pull/1)
publishes the numerical and input corrections without a dependency on this
repository's quality engine. Its default retains the original reaction cohort.

After installing `requirements-research.txt` and activating the combined
environment, run the portable comparison from the combined repository root,
using locally available saved market data:

```text
python research/colonylab/scripts/reproduce_team_code_corrections.py --workers 2
```

An optional `--original-dir <checkout>` accepts an external original checkout
only when its used files match the pinned hashes after LF normalization. The historical
`run_team_code_corrections.py` has one import adaptation: its default baseline
location now points to the shipped `qam_p113_baseline/` fixture. Its scientific
calculations are preserved. The portable wrapper additionally verifies the
pinned baseline and any existing output source snapshot before reuse; no
external checkout is needed. Default data and output locations are relative
to the nested ColonyLab project. A fresh run records its actual source bytes
and paths; historical audit hashes remain evidence for their original runs.

The tracked [results bundle](../results/README.md) shares aggregate findings and
sanitized audit snapshots. Raw data and full local output directories remain
outside Git. Copying old audited output JSON to a different workspace does not
relocate its absolute provenance paths; those historical files are not silently
rewritten. Fresh runs write provenance for their actual local inputs.
