# -*- coding: utf-8 -*-
"""ECG AIP core functions
Español:
Este módulo contiene las funciones principales para el procesamiento de señales de ECG,
incluyendo la carga de datos de ECG, la eliminación de componentes de CC, la detección de picos R,
el trazado de regiones de ECG, la visualización detallada de ECG,
el cálculo de la matriz de confusión y el cálculo de métricas de rendimiento.

English:
This module contains the core functions for ECG signal processing, including loading ECG data,
removing DC components, detecting R peaks, plotting ECG regions, detailed ECG visualization,
confusion matrix calculation, and performance metrics computation.
"""

import os

import numpy as np
import scipy.signal as sig
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.widgets as msp
import matplotlib.gridspec as mgr
import wfdb

def Cargar_Ecg(nombre_archivo, val = None, lead = None,
               qrs_detections = None):
    """Carga señal ECG desde un archivo .mat o registro WFDB (.hea/.dat).

    Parameters
    ----------
    nombre_archivo : str
        Ruta del archivo a cargar (.mat o WFDB).
    val : str, optional
        Nombre de la variable dentro del .mat que contiene la señal (requerido para .mat).
    lead : int, optional
        Índice de la derivación a extraer.
    qrs_detections : str, optional
        Nombre de la variable en el .mat para los picos reales detectados.

    Returns
    -------
    tuple
        (ecg_one_lead, picos_reales, cant_muestras)
    """
    file_path = os.path.abspath(nombre_archivo)
    base, ext = os.path.splitext(file_path)
    ext = ext.lower()

    picos_reales = None

    if ext == '.mat':
        if val is None:
            raise ValueError("Archivo .mat detectado pero no se especificó 'val'.")

        mat_struct = sio.loadmat(file_path)

        if val not in mat_struct:
            raise ValueError(f"La variable '{val}' no existe dentro del archivo .mat.")

        # Extracción; algunos .mat contienen columnas o filas
        sig_data = np.asarray(mat_struct[val])

        if lead is not None:
            ecg_one_lead = np.asarray(sig_data[lead, :]).flatten()
        else:
            ecg_one_lead = sig_data.flatten()

        if qrs_detections is not None and qrs_detections in mat_struct:
            picos_reales = np.asarray(mat_struct[qrs_detections]).flatten()

    elif ext in ['.hea', '.dat']:
        # Para WFDB, pasar nombre sin extensión y rdsamp requiere el nombre del registro
        record_name = os.path.splitext(os.path.basename(file_path))[0]
        folder = os.path.dirname(file_path)
        record_path = os.path.join(folder, record_name)

        ecg_signal, fields = wfdb.rdsamp(record_path)
        if lead is not None:
            ecg_one_lead = ecg_signal[:, lead]
        else:
            ecg_one_lead = ecg_signal[:, 0]

    else:
        raise ValueError('Formato no soportado. Use .mat o WFDB (.hea/.dat).')

    cant_muestras = len(ecg_one_lead)

    return np.asarray(ecg_one_lead), picos_reales, cant_muestras


def Removedor_DC(ecg, D= 64, N= 20, window_length = 101,
                 polyorder = 9):
    """Remueve la componente DC de una señal ECG y realiza suavizado.

    Esta función aplica un filtro FIR/IIR diseñado con coeficientes construidos
    manualmente (Numerador y Denominador) para eliminar la componente en CC de la señal.
    Después se aplica un filtro de Savitzky-Golay para suavizar la señal.

    Parameters
    ----------
    ecg : np.ndarray
        Señal ECG de una sola derivación (1D).
    D : int, optional
        Factor de separación de coeficientes para el diseño del filtro (por defecto 64).
    N : int, optional
        Parámetro que define la longitud del denominador y posición de retardos (por defecto 20).
    window_length : int
        Longitud de ventana para el filtro Savitzky-Golay (debe ser impar) (por defecto 101).
    polyorder : int
        Orden del polinomio para Savitzky-Golay (por defecto 9).

    Returns
    -------
    np.ndarray
        Señal ECG filtrada y suavizada, con la misma longitud que la señal de entrada.

    Notes
    -----
    El diseño de los coeficientes Num/Den es específico para la supresión de CC
    y puede necesitar ajuste para otras señales o frecuencia de muestreo.
    """
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

    delay = D*N
    ecg_padded = np.pad(ecg, (delay, delay), mode='edge')
    ECG_filtrado = sig.lfilter(Num, Den, ecg_padded)
    ECG_Golay = sig.savgol_filter(ECG_filtrado, window_length, polyorder)

    # Recorte final
    ECG_Golay = ECG_Golay[2*delay : 2*delay + len(ecg)]

    return ECG_Golay


def Detectar_picos_R_AIP(ecg, fs,
                         percentile = 30,
                         trgt_width = 0.06,
                         trgt_min_pattern_separation = 0.3):
    """Detección de picos R basada en un detector tipo AIP (impulsivo pseudoperiódico).

    El detector crea un patrón gaussiano derivado como plantilla, aplica filtrado
    para resaltar las subidas de las ondas R, suaviza la señal y detecta picos
    mediante un umbral basado en percentiles y una separación mínima entre picos.

    Parameters
    ----------
    ecg : np.ndarray
        Señal ECG en amplitud (1D).
    fs : float
        Frecuencia de muestreo en Hz.
    percentile : float, optional
        Percentil usado para establecer el umbral de detección en la señal procesada (por defecto 30).
    trgt_width : float, optional
        Ancho objetivo de la forma del pulso (en segundos) que se utiliza para diseñar el patrón (por defecto 0.06 s).
    trgt_min_pattern_separation : float, optional
        Separación mínima entre patrones detectados (en segundos) para evitar detección múltiple de un mismo latido (por defecto 0.3 s).

    Returns
    -------
    np.ndarray
        Indices (enteros) de las muestras donde se detectaron picos R.

    Notes
    -----
    - El valor de `percentile` y `trgt_min_pattern_separation` puede requerir ajuste para distintos registros.
    """
    pattern_size = 2 * round(trgt_width / 2 * fs) + 1

    g = sig.windows.gaussian(pattern_size + 1, std=pattern_size / 6)
    pattern_coeffs = np.diff(g) * sig.windows.gaussian(pattern_size, std=pattern_size / 6)

    rise_detector = sig.filtfilt(pattern_coeffs, 1, ecg)
    lp_size = round(1.2 * pattern_size)
    rise_detector = sig.filtfilt(np.ones(lp_size)/lp_size, 1, np.abs(rise_detector))
    rise_detector = sig.filtfilt(np.ones(lp_size)/lp_size, 1, rise_detector)

    thr = np.percentile(rise_detector, percentile)

    min_distance = int(trgt_min_pattern_separation * fs)
    peaks_R, _ = sig.find_peaks(rise_detector, height=thr, distance=min_distance)

    peaks_R_True = []
    window_half = int(0.05 * fs)

    for pk in peaks_R:
        i1 = max(0, pk - window_half)
        i2 = min(len(ecg), pk + window_half)
        local_peak = i1 + np.argmax(ecg[i1:i2])
        peaks_R_True.append(local_peak)

    return np.array(peaks_R_True, dtype=int)


def Graficar_regiones_ecg(ecg_one_lead_raw, ecg_one_lead_filter,
                          peaks_R, regs_interes, fs_ECG= 1000,
                          fig_sz_x = 10, fig_sz_y= 7, fig_dpi = 100):
    """Grafica regiones de interés (ventanas) del ECG en una figura por región.

    Para cada región de `regs_interes` se genera una figura con la señal original y
    la señal filtrada, además de marcar los picos R y el número de latidos por minuto (BPM)
    calculado dentro de esa ventana.

    Parameters
    ----------
    ecg_one_lead_raw : np.ndarray
        Señal ECG original (sin filtrar).
    ecg_one_lead_filter : np.ndarray
        Señal ECG filtrada o procesada que se desea visualizar.
    peaks_R : np.ndarray
        Índices de picos R detectados en la señal filtrada.
    regs_interes : iterable
        Lista o array de regiones de interés. Cada elemento debe contener un par [inicio, fin]
        en número de muestras que definan la ventana a plotear.
    fs_ECG : float, optional
        Frecuencia de muestreo en Hz. Se usa para calcular BPM (por defecto 1000).
    fig_sz_x, fig_sz_y : int, optional
        Tamaño (anchura, altura) de la figura en pulgadas (por defecto 10x7).
    fig_dpi : int, optional
        DPI de la figura (por defecto 100).

    Returns
    -------
    None
        Esta función solo muestra las figuras mediante Matplotlib (no devuelve objetos).
    """
    cant_muestras = len(ecg_one_lead_filter)

    for ii in regs_interes:
        ii = np.array(ii, dtype=float)
        zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')

        plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi=fig_dpi, facecolor='w')
        plt.plot(zoom_region, ecg_one_lead_raw[zoom_region], label='ECG original', linewidth=1)
        plt.plot(zoom_region, ecg_one_lead_filter[zoom_region], label='ECG filtrado ', linewidth=2)

        peaks_R_in_region = [p for p in peaks_R if p >= zoom_region[0] and p <= zoom_region[-1]]

        duracion_seg = len(zoom_region) / fs_ECG
        bpm = len(peaks_R_in_region) * 60 / duracion_seg if duracion_seg > 0 else np.nan

        plt.plot(peaks_R_in_region, ecg_one_lead_filter[peaks_R_in_region], 'ro',
                 label=f'Picos R ({bpm:.1f} BPM)')

        plt.title(f'ECG desde {ii[0]:.0f} hasta {ii[1]:.0f} muestras')
        plt.ylabel('Amplitud (adimensional)')
        plt.xlabel('Muestras (#)')

        axes_hdl = plt.gca()
        axes_hdl.legend()
        axes_hdl.set_yticks(())

        plt.show()


def Graficar_ecg_detallado(ecg, peaks_R, fs, time = None):
    """Visualización detallada e interactiva del ECG.

    Se presenta una figura con tres subplots: el ECG completo, una ventana ampliada
    seleccionable (mediante SpanSelector) y un gráfico de Poincaré calculado sobre
    los intervalos RR depositados en `peaks_R`.

    Parameters
    ----------
    ecg : np.ndarray
        Señal ECG completa.
    peaks_R : np.ndarray
        Índices de las muestras con picos R detectados.
    fs : float
        Frecuencia de muestreo en Hz.
    time : np.ndarray, optional
        Vector de tiempos en segundos. Si se omite, se genera como np.arange(len(ecg))/fs.

    Returns
    -------
    None
        La función muestra la figura interactiva con Matplotlib.
    """
    if time is None:
        time = np.arange(len(ecg)) / fs

    fig = plt.figure(figsize=(14, 7))
    gs = mgr.GridSpec(2, 2, width_ratios=[2.5, 1])
    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_poincare = fig.add_subplot(gs[:, 1])

    fig.suptitle("Visualización detallada del ECG con Poincaré", fontsize=16)

    ax_full.plot(time, ecg, color='gold')
    if peaks_R is not None:
        ax_full.plot(time[peaks_R], ecg[peaks_R], 'ro', label='Picos R')

    ax_full.set_title("ECG completo")
    ax_full.set_ylabel("Amplitud [mV]")
    ax_full.grid(True)

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

    fig.span = msp.SpanSelector(ax_full, onselect, 'horizontal', useblit=True,
                                props=dict(alpha=0.3, facecolor='red'))

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


def Matriz_De_Confusion(peaks_R, true_peaks, tol, cant_muestras):
    """Computa matriz de confusión y índices de coincidencia entre detecciones y verdades.

    Este método compara vectores de picos detectados (`peaks_R`) con picos reales
    (`true_peaks`) usando una tolerancia `tol` en muestras. Para cada pico detectado,
    se asigna como verdadero positivo (VP) si existe un pico real no emparejado dentro
    de la tolerancia; de lo contrario se marca como falso positivo (FP). Los picos
    reales no emparejados son falsos negativos (FN). Los verdaderos negativos (VN)
    se cuentan como el resto de las muestras que no contienen eventos.

    Parameters
    ----------
    peaks_R : np.ndarray
        Índices de picos detectados.
    true_peaks : np.ndarray
        Índices de picos reales (ground truth).
    tol : int
        Tolerancia en número de muestras para considerar una detección como correcta.
    cant_muestras : int
        Cantidad total de muestras de la señal (usado para calcular VN).

    Returns
    -------
    tuple
        (VP_idx, VN_count, FP_idx, FN_idx, confusion_matrix)
        - VP_idx: lista de índices de verdaderos positivos (detecciones correctas)
        - VN_count: número de verdaderos negativos (muestras sin evento)
        - FP_idx: lista de índices de falsos positivos (detecciones incorrectas)
        - FN_idx: lista de índices de falsos negativos (verdaderos picos no detectados)
        - confusion_matrix: matriz 2x2 en orden [[TP, FP], [FN, TN]]

    Notes
    -----
    - La matriz de confusión resultante está en el formato clásico: filas=predicción, columnas=ground truth.
    """
    VP_idx = []
    VN_idx = []
    FP_idx = []
    FN_idx = []

    matched = np.zeros(len(true_peaks), dtype=bool)

    for p in peaks_R:
        diffs = np.abs(true_peaks - p)
        min_dist = np.min(diffs)
        idx_min = np.argmin(diffs)

        if min_dist <= tol and not matched[idx_min]:
            VP_idx.append(p)
            matched[idx_min] = True
        else:
            FP_idx.append(p)

    FN_idx = list(true_peaks[~matched])
    VN_count = cant_muestras - (len(VP_idx) + len(FP_idx) + len(FN_idx))

    confusion_matrix = np.array([[len(VP_idx), len(FP_idx)],
                                 [len(FN_idx), VN_count]])

    return VP_idx, VN_count, FP_idx, FN_idx, confusion_matrix


def Metricas(conf_matrix):
    """Calcula las métricas básicas de clasificación a partir de una matriz de confusión.

    La función devuelve precision, recall, f1 y accuracy calculadas a partir
    de una matriz 2x2 en el formato [[TP, FP], [FN, TN]].

    Parameters
    ----------
    conf_matrix : np.ndarray
        Matriz de confusión 2x2 (numérica). Debe tener la forma [[TP, FP], [FN, TN]].

    Returns
    -------
    tuple
        (precision, recall, f1, accuracy) — métricas flotantes.

    Notes
    -----
    - Se añade protección contra división por cero: si el denominador es 0, la métrica se devuelve como 0.
    """
    TP, FP, FN, TN = conf_matrix.ravel()

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (TP + TN) / np.sum(conf_matrix) if np.sum(conf_matrix) else 0

    return precision, recall, f1, accuracy
