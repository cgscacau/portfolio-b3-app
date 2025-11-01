"""
Core module - Business logic para análise de portfólios B3
"""

__version__ = "1.0.0"
__author__ = "Portfolio B3 Team"

from . import data
from . import filters
from . import metrics
from . import opt
from . import ui

__all__ = ['data', 'filters', 'metrics', 'opt', 'ui']
