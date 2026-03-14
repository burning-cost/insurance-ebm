import warnings

warnings.warn(
    "insurance-ebm is deprecated. Install insurance-gam instead: pip install insurance-gam",
    DeprecationWarning,
    stacklevel=2,
)

from insurance_gam.ebm import *  # noqa: F401, F403, E402
