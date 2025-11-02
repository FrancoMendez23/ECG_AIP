# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:50:05 2025

@author: Usuario
"""
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
from scipy.signal import find_peaks, windows, filtfilt
from matplotlib.widgets import SpanSelector
from matplotlib.gridspec import GridSpec

plt.close("all")

def Cargar_Ecg(nombre_archivo):
    """
    Carga un archivo .mat con la variable 'ecg_lead' y devuelve la señal y su longitud.

    Parameters
    -----------
    nombre_archivo : str
                    Nombre del archivo .mat (por ejemplo 'ecg.mat')

    Returns
    --------
    ecg_one_lead : np.ndarray
                    Vector con la señal ECG (aplanado)
    cant_muestras : int
                    Cantidad de muestras del vector
    """
    fig_sz_x = 10
    fig_sz_y = 7
    fig_font_size = 16
    
    mpl.rcParams['figure.figsize'] = (fig_sz_x,fig_sz_y)
    plt.rcParams.update({'font.size':fig_font_size})
    
    mat_struct = sio.loadmat(nombre_archivo)
    
    if 'ecg_lead' not in mat_struct:
        raise KeyError('El archivo no contiene la variable "ecg_lead"')
        
    # Extraer  la señal
    ecg_one_lead = mat_struct['ecg_lead'].flatten()
    
    # Extraer picos reales
    picos_reales = mat_struct['qrs_detections'].flatten()
    
    #Cantidad de Muestras
    cant_muestras = len(ecg_one_lead)

    return ecg_one_lead, picos_reales,cant_muestras
def Removedor_DC(ecg, D=64, N=20, window_length=101, polyorder=9):
    """
    Aplica un filtro de removedor de DC de Fase lineal
    Y tambien un suavizado Savitzky-Golay a la señal ECG.
    
    https://www.dsprelated.com/showarticle/58.php
    
    Parameters:
    -----------
    ecg_one_lead : np.ndarray
        Señal ECG de una sola derivación.
    D : int
        Parámetro de retardo para el cálculo del filtro (default 64).
    N : int
        Orden del filtro (default 20).
    window_length : int
        Tamaño de la ventana del filtro Savitzky-Golay (debe ser impar).
    polyorder : int
        rden del polinomio del filtro Savitzky-Golay.

    Return:
    --------
    ECG_Golay : np.ndarray
        Señal ECG filtrada y suavizada.
        
    """

    # ---- Filtro principal ----
    Num = np.zeros(2*D*N + 1)
    Den = np.zeros(2*N + 1)

    # Numerador
    Num[0] = -1 / D**2
    Num[D*N - N] = 1
    Num[D*N] = (2 / D**2 - 2)
    Num[D*N + N] = 1
    Num[2*D*N] = -1 / D**2

    # Denominador
    Den[0] = 1
    Den[N] = -2
    Den[2*N] = 1

    # ---- Aplicar filtro ----
    ECG_filtrado = sig.lfilter(Num, Den, ecg)

    # ---- Suavizado Savitzky-Golay ----
    ECG_Golay = sig.savgol_filter(ECG_filtrado, window_length, polyorder)
    
    # ---- Quita el retardo ----
    delay_pos =  int(2/2* (D - 1) * N)
    ECG_Golay = np.roll(ECG_Golay, -delay_pos)

    return ECG_Golay
def Detectar_picos_R_AIP(ecg, fs, trgt_width=0.06, trgt_min_pattern_separation=0.3):
    """
    Detección de picos R basada en un detector tipo AIP (impulsivo pseudoperiódico).
    
    Parámetros:
    -----------
    ecg : array
        Señal ECG (ya filtrada).
    fs : float
        Frecuencia de muestreo en Hz.
    trgt_width : float
        Duración aproximada del complejo QRS en segundos.
    trgt_min_pattern_separation : float
        Tiempo mínimo entre dos picos R consecutivos, en segundos.

    Retorna:
    --------
    peaks_R_True : array
        Índices (muestras) donde se detectaron los picos R.
    """

    # Tamaño del patrón (ventana)
    pattern_size = 2 * round(trgt_width / 2 * fs) + 1  # forzar impar

    # Coeficientes del patrón tipo derivada de gaussiana
    g = windows.gaussian(pattern_size + 1, std=pattern_size / 6)
    pattern_coeffs = np.diff(g) * windows.gaussian(pattern_size, std=pattern_size / 6)

    # "Rise detector" 
    rise_detector = filtfilt(pattern_coeffs, 1, ecg)
    lp_size = round(1.2 * pattern_size)
    rise_detector = filtfilt(np.ones(lp_size)/lp_size, 1, np.abs(rise_detector))
    rise_detector = filtfilt(np.ones(lp_size)/lp_size, 1, rise_detector)
    
    thr = np.percentile(rise_detector, 30)

    # Detectar máximos locales que superen el umbral
    min_distance = int(trgt_min_pattern_separation * fs)
    peaks_R, _ = find_peaks(rise_detector, height=thr, distance=min_distance)
    
    # Ventana para detectar picos R
    peaks_R_True = []
    window_half = int(0.05 * fs)  

    for pk in peaks_R:
        i1 = max(0, pk - window_half)
        i2 = min(len(ecg), pk + window_half)
    
        local_peak = i1 + np.argmax(ecg[i1:i2])
        peaks_R_True.append(local_peak)

    peaks_R_True = np.array(peaks_R_True)
    
    return peaks_R_True
def Graficar_regiones_ecg(ecg_one_lead_raw, ecg_one_lead_filter, peaks_R, 
                          regs_interes, fs_ECG=1000,
                          fig_sz_x=10, fig_sz_y=7, fig_dpi=100):
    """
    Grafica regiones de interés de una señal ECG con los picos R detectados.

    Parámetros:
    -----------
    ecg_one_lead_raw : np.ndarray
        Señal ECG original.
    ecg_one_lead_filter : np.ndarray
        Señal filtrada con Savitzky-Golay.
    peaks_R: list o np.ndarray
        Índices de los picos R, T y P respectivamente.
    regs_interes : iterable
        Lista o tupla de regiones de interés. Cada región = [inicio, fin] (en muestras).
        Puede contener también np.array con tiempos en minutos * 60 * Fs.
    fs_ECG : float
        Frecuencia de muestreo [Hz].
    fig_sz_x, fig_sz_y, fig_dpi : parámetros de figura.
    """

    cant_muestras = len(ecg_one_lead_filter)

    for ii in regs_interes:
        # Convertir a np.array por si es lista
        ii = np.array(ii, dtype=float)

        # Asegurar límites dentro del rango de la señal
        zoom_region = np.arange(
            np.max([0, ii[0]]),
            np.min([cant_muestras, ii[1]]),
            dtype='uint'
        )

        # Crear figura
        plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi=fig_dpi, facecolor='w', edgecolor='k')
        plt.plot(zoom_region, ecg_one_lead_raw[zoom_region], label='ECG original', linewidth=1)
        plt.plot(zoom_region, ecg_one_lead_filter[zoom_region], label='ECG filtrado ', linewidth=2)

        # Filtrar picos dentro de la región
        peaks_R_in_region = [p for p in peaks_R if p >= zoom_region[0] and p <= zoom_region[-1]]

        # Calcular duración y BPM en la región
        duracion_seg = len(zoom_region) / fs_ECG
        bpm = len(peaks_R_in_region) * 60 / duracion_seg if duracion_seg > 0 else np.nan

        # Graficar los picos
        plt.plot(peaks_R_in_region, ecg_one_lead_filter[peaks_R_in_region], 'ro',
                 label=f'Picos R ({bpm:.1f} BPM)')

        # Etiquetas y leyenda
        plt.title(f'ECG desde {ii[0]:.0f} hasta {ii[1]:.0f} muestras')
        plt.ylabel('Amplitud (adimensional)')
        plt.xlabel('Muestras (#)')

        axes_hdl = plt.gca()
        axes_hdl.legend()
        axes_hdl.set_yticks(())

        plt.show()
def Graficar_ecg_detallado(ecg,peaks_R,fs,time=None):
    """
    Visualiza una señal ECG con una vista general (arriba)
    y una vista detallada (abajo) interactiva.
    
    Parámetros
    ----------
    ecg : array_like
          Señal de ECG.
    peaks_R : array_like
          Indices de los picos R  
    fs : float
          Frecuencia de muestreo (Hz).
    time : array_like, opcional
          Vector de tiempo (si no se pasa, se genera automáticamente).
    """

    if time is None:
        time = np.arange(len(ecg)) / fs
  
    # Preparar la figura con 3 subplots: completo, zoom y Poincaré
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, width_ratios=[2.5, 1])  # más ancho el ECG que el Poincaré
    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_poincare = fig.add_subplot(gs[:, 1])  # ocupa las dos filas (derecha)

    fig.suptitle("Visualización detallada del ECG con Poincaré", fontsize=16)
    
    # Señal completa
    #ax_full.plot(time, ecg, color='gold') Si quiero ver la señal, descomentar
    if peaks_R is not None:
        ax_full.plot(time[peaks_R], ecg[peaks_R], 'ro', label='Picos R')
    ax_full.set_title("ECG completo")
    ax_full.set_ylabel("Amplitud [mV]")
    ax_full.grid(True)

    # Vista ampliada inicial (por defecto, primeros 3 segundos)
    init_end = min(3, time[-1])
    mask = (time >= 0) & (time <= init_end)
    ax_zoom.plot(time[mask], ecg[mask], color='gold')
    if peaks_R is not None:
        zoom_peaks = [i for i in peaks_R if 0 <= i < len(time) and 0 <= time[i] <= init_end]
        ax_zoom.plot(time[zoom_peaks], ecg[zoom_peaks], 'ro')
    ax_zoom.set_xlim(0, init_end)
    ax_zoom.set_title("Ventana seleccionada")
    ax_zoom.set_xlabel("Tiempo [s]")
    ax_zoom.set_ylabel("Amplitud [mV]")
    ax_zoom.grid(True)

  # Callback cuando seleccionás una región
    def onselect(xmin, xmax):
      indmin, indmax = np.searchsorted(time, (xmin, xmax))
      indmax = min(len(time), indmax)
      region_t = time[indmin:indmax]
      region_ecg = ecg[indmin:indmax]
      
      ax_zoom.clear()
      ax_zoom.plot(region_t, region_ecg, color='gold')
      if peaks_R is not None:
           zoom_peaks = [i for i in peaks_R if indmin <= i < indmax]
           ax_zoom.plot(time[zoom_peaks], ecg[zoom_peaks], 'ro')
      ax_zoom.set_xlim(region_t[0], region_t[-1])
      ax_zoom.set_title(f"Ventana seleccionada: {xmin:.2f}s – {xmax:.2f}s")
      ax_zoom.set_xlabel("Tiempo [s]")
      ax_zoom.set_ylabel("Amplitud [mV]")
      ax_zoom.grid(True)
      fig.canvas.draw_idle()

  # SpanSelector (selección con el mouse)
    fig.span = SpanSelector(
        ax_full, onselect, 'horizontal',
        useblit=True,
        props=dict(alpha=0.3, facecolor='red')
    )
    # --- Poincaré ---
    if peaks_R is not None and len(peaks_R) > 1:
        RR_intervals = np.diff(peaks_R) / fs
        ax_poincare.scatter(RR_intervals[:-1], RR_intervals[1:], color='red', alpha=0.2, edgecolor='k')
        ax_poincare.set_title('Gráfico de Poincaré del ECG', fontsize=14)
        ax_poincare.set_xlabel(r'$RR_n$ [s]', fontsize=12)
        ax_poincare.set_ylabel(r'$RR_{n+1}$ [s]', fontsize=12)
        ax_poincare.grid(True)
        ax_poincare.axis('equal')
    
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def Matriz_De_Confusion(peaks_R,true_peaks,tol,cant_muestras):
    """
   Calcula VP, VN, FP y FN 
   
   Parámetros:
   -----------
   peaks_R :  np.ndarray
       Índices predecidos
   true_peaks :  np.ndarray
       Índices verdaderos (ground truth)
   tol : int
       Tolerancia máxima (en muestras) para considerar una coincidencia como TP
   
   Retorna:
   --------
   VP_idx : lista
       Índices de verdaderos positivos
   VN_idx : lista
       Índices de verdaderos negativos       
   FP_idx : lista
       Índices de falsos positivos
   FN_idx : lista
       Índices de falsos negativos
   """
    VP_idx = []
    VN_idx = []
    FP_idx = []
    FN_idx = []

    matched = np.zeros(len(true_peaks), dtype=bool)

    for p in peaks_R:
        # calcular la distancia mínima a un real
        diffs = np.abs(true_peaks - p)
        min_dist = np.min(diffs)
        idx_min = np.argmin(diffs)

        if min_dist <= tol and not matched[idx_min]:
            VP_idx.append(p)
            matched[idx_min] = True
        else:
            FP_idx.append(p)

    # FN = reales que no fueron emparejados
    FN_idx = list(true_peaks[~matched])

    VN_idx = cant_muestras - (len(VP_idx) + len(FP_idx) + len(FN_idx))
    
    #Matriz de confusion
    confusion_matrix = np.array([[len(VP_idx), len(FP_idx)],
                                 [len(FN_idx), VN_idx]])
    
    return VP_idx, VN_idx, FP_idx,FN_idx, confusion_matrix
def Metricas(conf_matrix):
    """
    Calcula precisión, recall y F1 a partir de la matriz de confusión.
    
    """
    VN, FP, FN, VP = conf_matrix.ravel()
    precision = VP / (VP + FP) if (VP + FP) else 0
    recall = VP / (VP + FN) if (VP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (VP + VN) / np.sum(conf_matrix) if np.sum(conf_matrix) else 0
    
    return precision, recall, f1, accuracy

ecg_one_lead, picos_reales, cant_muestras = Cargar_Ecg('ecg.mat')
ecg_golay = Removedor_DC(ecg_one_lead, D=64, N=20, window_length=101, polyorder=9)
peaks_R_AIP = Detectar_picos_R_AIP(ecg_golay, fs=1000, trgt_width=0.09, trgt_min_pattern_separation=0.3)
Graficar_ecg_detallado(ecg_golay,peaks_R_AIP,fs=1000,time=None)
VP, VN, FP, FN, conf = Matriz_De_Confusion(peaks_R_AIP, picos_reales, tol=10,cant_muestras = cant_muestras)
precision, recall, f1, acc = Metricas(conf)


