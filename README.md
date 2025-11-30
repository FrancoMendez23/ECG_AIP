# ♥️ Detector de Picos R en ECG — AIP 

Este repositorio contiene un pipeline completo para:

- Cargar señales ECG desde archivos `.mat`
- Eliminar componente DC y suavizar la señal mediante un filtro lineal + Savitzky-Golay
- Detectar picos R usando un detector AIP (Impulsivo Pseudoperiódico)
- Visualizar el ECG con zoom interactivo + gráfico de Poincaré
- Evaluar la detección mediante matriz de confusión y métricas

---

## 📌 Contenido

El archivo principal (`Deteccion_AIP.py`) incluye:

### ✔ `Cargar_Ecg`
Carga señales ECG desde un archivo `.mat`o`.dat`, permitiendo seleccionar derivaciones y extraer picos reales si se incluyen en el dataset.

### ✔ `Removedor_DC`
Filtrado de componente DC basado en un filtro lineal de fase lineal de Rick Lions + suavizado Savitzky-Golay.

### ✔ `Detectar_picos_R_AIP`
Detector inspirado en patrones impulsivos pseudoperiódicos:

- Derivada de Gaussiana  
- Filtrado bidireccional (`filtfilt`)  
- Rise detector + umbral por percentil  


### ✔ `Graficar_ecg_detallado`
Interfaz gráfica interactiva con:

- ECG completo  
- ventana ampliada seleccionable con el mouse  
- Gráfico de Poincaré (RRₙ vs RRₙ₊₁)

### ✔ `Matriz_De_Confusion`
Calcula VP, FP, FN, VN con tolerancia configurable.

### ✔ `Metricas`
Calcula precisión, recall, F1-score y accuracy.

---

## 🚀 Ejemplo de uso

```python
ecg_one_lead, picos_reales, cant_muestras = Cargar_Ecg(
    'ecg.mat', 'ecg_lead', qrs_detections='qrs_detections'
)

ecg_golay = Removedor_DC(
    ecg_one_lead, D=64, N=20,
    window_length=101, polyorder=9
)

peaks_R_AIP = Detectar_picos_R_AIP(
    ecg_golay, fs=1000,
    percentile=30,
    trgt_width=0.09,
    trgt_min_pattern_separation=0.3
)

Graficar_ecg_detallado(ecg_golay, peaks_R_AIP, fs=1000)

VP, VN, FP, FN, conf = Matriz_De_Confusion(
    peaks_R_AIP, picos_reales, tol=30,
    cant_muestras=cant_muestras
)

precision, recall, f1, acc = Metricas(conf)
