⚠️ **This package has been merged into [`insurance-gam`](https://github.com/burning-cost/insurance-gam).** This repository is archived. Install `insurance-gam` instead.

# insurance-ebm — Deprecated

This package has been superseded by [insurance-gam](https://github.com/burning-cost/insurance-gam).

All functionality from insurance-ebm — the `InsuranceEBM` wrapper, `RelativitiesTable`, `Diagnostics`, `MonotonicityEditor`, and `GLMComparison` — is now part of insurance-gam under the `insurance_gam.ebm` subpackage. The source code was merged verbatim.

## Migration

```bash
pip install insurance-gam
```

```python
# Before
from insurance_ebm import InsuranceEBM, RelativitiesTable

# After
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable
```

This repository is archived and will not receive further updates.
