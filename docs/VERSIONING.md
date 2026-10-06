# Release versioning

Published version tags are immutable. Keep old images and digests for rollback.

- Patch (1.0.x): compatible fixes and internal optimizations; serving contract unchanged.
- Minor (1.x.0): new supported features or deployment profiles.
- Major (x.0.0): breaking serving or deployment changes.
- Release candidates (-rc.1, -rc.2): test builds, not qualified stable releases.

Image variants retain the suffix `-sm120-amd64-cu130`. The suffix identifies
architecture/platform/CUDA, not release maturity.

The FP8 QSA indexer candidate is `v1.0.1-rc.1-sm120-amd64-cu130`.
Promote the exact qualified digest to `v1.0.1-sm120-amd64-cu130`; do not rebuild
between final qualification and promotion. Record each patch revision,
backport diff and qualification results. Keep `v1.0.0` unchanged.

Candidate scope: only the FP8 QSA indexer backport (#54890). The main attention
KV dtype, MTP3, deterministic SM120 top-k and chat template remain unchanged.
FP8 indexer is selected explicitly with `--attention-config '{"indexer_kv_dtype":"fp8"}'`;
its default remains BF16. This document does not promote the candidate or
change the stable launch defaults.
