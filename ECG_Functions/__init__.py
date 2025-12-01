"""ECG AIP Functions Package
This package provides tools for processing and analyzing ECG signals.
"""
__author__ = 'Franco Mendez'
__version__ = '0.1.0'

from .core import (
    Cargar_Ecg,
    Removedor_DC,
    Detectar_picos_R_AIP,
    Graficar_regiones_ecg,
    Graficar_ecg_detallado,
    Matriz_De_Confusion,
    Metricas,
)

__all__ = [
    'Cargar_Ecg',
    'Removedor_DC',
    'Detectar_picos_R_AIP',
    'Graficar_regiones_ecg',
    'Graficar_ecg_detallado',
    'Matriz_De_Confusion',
    'Metricas',
]
