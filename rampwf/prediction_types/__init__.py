from .clustering import make_clustering
from .combined import make_combined
from .detection import make_detection
from .multiclass import make_multiclass
from .regression import make_regression
from .generative_regression import make_generative_regression

__all__ = [
    'make_clustering',
    'make_combined',
    'make_detection',
    'make_generative_regression',
    'make_multiclass',
    'make_regression',
]
