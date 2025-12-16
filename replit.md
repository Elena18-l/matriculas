# Detector y Segmentador de Matrículas

## Descripción
Aplicación Python/Streamlit que detecta matrículas de vehículos en imágenes y segmenta sus componentes sin necesidad de OCR.

## Funcionalidades
- Carga de imágenes mediante drag-and-drop o selección de archivo
- Detección automática de matrículas usando OpenCV (detección por contornos)
- Segmentación en tres zonas:
  - **Banda de país** (azul): Zona izquierda típica de matrículas europeas
  - **Números** (rojo): 4 dígitos en formato español
  - **Letras** (verde): 3 letras en formato español
- Visualización con bounding boxes coloreados
- Soporte para múltiples matrículas por imagen (hasta 5)

## Stack Técnico
- **Framework**: Streamlit
- **Procesamiento de imagen**: OpenCV (cv2)
- **Dependencias**: numpy, pillow, opencv-python-headless

## Ejecución
```bash
streamlit run app.py --server.port 5000
```

## Estructura
- `app.py`: Aplicación principal con toda la lógica de detección y UI
- `.streamlit/config.toml`: Configuración del servidor Streamlit

## Algoritmo de Detección
1. Preprocesamiento: escala de grises y desenfoque gaussiano
2. Detección de bordes con Canny
3. Operaciones morfológicas para cerrar contornos
4. Filtrado por relación de aspecto (2:1 a 6:1) típica de matrículas
5. Segmentación de caracteres por umbralización y contornos
