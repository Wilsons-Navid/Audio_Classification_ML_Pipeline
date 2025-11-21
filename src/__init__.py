"""
Audio Classification ML Pipeline

Main package for audio classification system
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "wadotiwawil@gmail.com"

from . import preprocessing
from . import model
from . import prediction
from . import utils

__all__ = ['preprocessing', 'model', 'prediction', 'utils']
