# ♥️ Detector de Picos R en ECG — AIP 

Este repositorio contiene un pipeline completo para:

- Cargar señales ECG desde archivos `.mat` o `.dat`
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

