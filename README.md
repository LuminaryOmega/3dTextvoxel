# 3dTextvoxel

This repository now includes a minimal Python implementation of the Earth-ECEF voxel
binding and guardian enforcement primitives outlined in the specification. The
core module lives in `textvoxel/binding.py` and focuses on deterministic
quantization, semantic canonicalization (S1 bytes), identity hashing, unknown
sigil handling, and guardian payload checks.

Known TODO boundaries:
- Deterministic ECEF conversion must be supplied by the caller; the module will
  raise a TODO marker until provided.
- Sigil output paths must be configured explicitly; defaults are not assumed.

Running tests:

```bash
python -m unittest discover -s tests -v
```
