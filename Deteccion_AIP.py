import argparse
import os

from ECG_Functions import (
    Cargar_Ecg,
    Removedor_DC,
    Select_Patron,
    Detect_Patron,
    Plot_Detector,
    Detectar_picos_R_AIP,
    Graficar_ecg_detallado,
    Matriz_De_Confusion,
    Metricas,
)

def main(file, varname, lead, fs):
    """Ejecuta un flujo básico: carga, preprocesado, detección, evaluación y graficado.

    Parameters
        ----------
        file : str
            Ruta del archivo a cargar (.mat o WFDB).
        varname : str, optional
            Nombre de la variable dentro del .mat que contiene la señal (requerido para .mat).
        lead : int, optional
            Índice de la derivación a extraer.
        fs : float, optional
            Frecuencia de muestreo en Hz.
    """
    ecg_one_lead, picos_reales, cant_muestras = Cargar_Ecg(
        file, val=varname, lead=lead, qrs_detections='qrs_detections'
    )

    ecg_golay = Removedor_DC(ecg_one_lead, D=64, N=7, window_length=19, polyorder=3)
    pattern = Select_Patron(ecg_golay, fs)
    peaks, detector,thr_min,thr_max, zone = Detect_Patron(ecg_golay,pattern,fs,percentile_min=20,trgt_width= 0.09*2,trgt_min_pattern_separation=0.4,percentile_max=None)
    Plot_Detector(ecg_golay,detector,fs,peaks,thr_min=thr_min,thr_max=thr_max)
    Graficar_ecg_detallado(ecg_golay, peaks, fs=fs,zone=zone)
    
    #peaks_R_AIP = Detectar_picos_R_AIP(ecg_golay, fs=fs, percentile=80, trgt_width=0.09, trgt_min_pattern_separation=0.4)
    #Graficar_ecg_detallado(ecg_golay, peaks_R_AIP, fs=fs, time=None)

    VP, VN, FP, FN, conf = Matriz_De_Confusion(peaks, picos_reales, tol=30, cant_muestras=cant_muestras)
    precision, recall, f1, acc = Metricas(conf)

    print('Precision: {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    print('F1-score: {:.4f}'.format(f1))
    print('Accuracy: {:.4f}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECG AIP detection')
    parser.add_argument('--file', '-f', default='ecg.mat', help='Archivo .mat o registro WFDB (.hea/.dat)')
    parser.add_argument('--varname', '-v', default='ecg_lead', help='Nombre de variable dentro de .mat')
    parser.add_argument('--lead', '-l', default=None, type=int, help='Derivacion a usar (int)')
    parser.add_argument('--fs', default=1000, type=float, help='Frecuencia de muestreo (Hz)')
    args = parser.parse_args()

    args.file = os.path.join(os.path.dirname(__file__), args.file)
    main(args.file, args.varname, args.lead, args.fs)
