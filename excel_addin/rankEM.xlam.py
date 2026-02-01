"""
xlwings Excel Add-in configuration file.

This file is used by xlwings to configure the Excel add-in.
The actual UDF implementations are in rankEM_functions.py.
"""

# Import all UDFs from rankEM_functions
from rankEM_functions import (
    RankEM_Theta,
    RankEM_Beta,
    RankEM_Imputed,
    RankEM_Stats,
    RankEM_Ranking,
    RankEM_AllMethods,
    RankEM_Version,
    RankEM_Help,
)
