"""Training utilities: walk-forward out-of-sample prediction."""

from berich.training.cross_sectional import CrossSectionalOof, oof_predict_cross_sectional
from berich.training.walk_forward import OofResult, oof_predict

# Note: ``tournament`` is intentionally NOT re-exported here — it imports the registry +
# calibration eagerly and would create an import cycle (tournament imports
# ``berich.training.oof_predict`` from this package). Import it as
# ``from berich.training.tournament import ...`` instead.

__all__ = [
    "CrossSectionalOof",
    "OofResult",
    "oof_predict",
    "oof_predict_cross_sectional",
]
