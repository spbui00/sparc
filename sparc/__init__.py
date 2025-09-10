from .core.base_method import BaseSACMethod
from .core.data_handler import NeuralDataHandler
from .core.evaluator import SPARCEvaluator
from .core.neural_analyzer import NeuralAnalyzer
from .methods.template_subtraction import (
    BaseTemplateSubtraction,
    AverageTemplateSubtraction,
)

__version__ = "0.0.1"
__all__ = [
    "BaseSACMethod",
    "NeuralDataHandler", 
    "SPARCEvaluator",
    "NeuralAnalyzer",
    "BaseTemplateSubtraction",
    "AverageTemplateSubtraction",
]
