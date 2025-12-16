import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


def preprocess_image(image):
    """Convert image to grayscale and apply preprocessing for plate detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blur


def detect_license_plates(image):
    """Detect potential license plate regions in the image."""
    gray, blur = preprocess_image(image)
    
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    img_height, img_width = image.shape[:2]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        img_area = img_width * img_height
        area_ratio = area / img_area
        
        if 2.0 < aspect_ratio < 6.0 and 0.005 < area_ratio < 0.15:
            if w > 60 and h > 15:
                plates.append((x, y, w, h))
    
    plates = sorted(plates, key=lambda p: p[2] * p[3], reverse=True)
    
    return plates[:5]


def segment_plate_regions(plate_img):
    """Segment the plate into country band, letters, and numbers regions."""
    h, w = plate_img.shape[:2]
    
    segments = {
        'country_band': None,
        'characters': [],
        'visualization': None
    }
    
    country_band_width = int(w * 0.12)
    if country_band_width > 5:
        segments['country_band'] = {
            'region': plate_img[:, :country_band_width],
            'bbox': (0, 0, country_band_width, h),
            'type': 'country'
        }
    
    main_plate = plate_img[:, country_band_width:] if country_band_width > 5 else plate_img
    main_w = main_plate.shape[1]
    
    gray = cv2.cvtColor(main_plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_contours = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / float(ch) if ch > 0 else 0
        
        if 0.1 < aspect_ratio < 1.5 and ch > h * 0.3 and ch < h * 0.95:
            if cw > 5 and cw < main_w * 0.3:
                char_contours.append((x + country_band_width, y, cw, ch))
    
    char_contours = sorted(char_contours, key=lambda c: c[0])
    
    for i, (x, y, cw, ch) in enumerate(char_contours):
        char_region = plate_img[y:y+ch, x:x+cw]
        
        relative_pos = (x - country_band_width) / main_w if main_w > 0 else 0
        
        # Spanish plate format: NNNN LLL (4 numbers + 3 letters)
        # Numbers are on the left (first ~55%), letters on the right
        if relative_pos < 0.55:
            char_type = 'number'
        else:
            char_type = 'letter'
        
        segments['characters'].append({
            'region': char_region,
            'bbox': (x, y, cw, ch),
            'type': char_type,
            'position': i
        })
    
    return segments


def draw_visualization(image, plates):
    """Draw bounding boxes on detected plates."""
    vis_image = image.copy()
    
    for i, (x, y, w, h) in enumerate(plates):
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(vis_image, f'Matricula {i+1}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return vis_image


def draw_plate_segmentation(plate_img, segments):
    """Draw colored regions on the plate showing segmentation."""
    vis_plate = plate_img.copy()
    
    if segments['country_band'] is not None:
        bbox = segments['country_band']['bbox']
        x, y, w, h = bbox
        overlay = vis_plate.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, vis_plate, 0.6, 0, vis_plate)
        cv2.rectangle(vis_plate, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    for char in segments['characters']:
        bbox = char['bbox']
        x, y, w, h = bbox
        
        if char['type'] == 'letter':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        overlay = vis_plate.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.3, vis_plate, 0.7, 0, vis_plate)
        cv2.rectangle(vis_plate, (x, y), (x + w, y + h), color, 2)
    
    return vis_plate


def main():
    st.set_page_config(
        page_title="Detector de Matrículas",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Detector y Segmentador de Matrículas")
    st.markdown("""
    Esta aplicación detecta matrículas en imágenes de coches y separa sus componentes:
    - 🔵 **Banda de país** (zona azul izquierda)
    - 🔴 **Números** (formato español: 4 dígitos a la izquierda)
    - 🟢 **Letras** (formato español: 3 letras a la derecha)
    """)
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Sube una imagen con coches",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Sube una imagen que contenga coches con matrículas visibles"
    )
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imagen Original")
            st.image(pil_image, use_container_width=True)
        
        with st.spinner("Detectando matrículas..."):
            plates = detect_license_plates(image)
        
        if plates:
            vis_image = draw_visualization(image, plates)
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("🔍 Matrículas Detectadas")
                st.image(vis_image_rgb, use_container_width=True)
            
            st.divider()
            st.subheader("📊 Análisis de Matrículas")
            
            for i, (x, y, w, h) in enumerate(plates):
                st.markdown(f"### Matrícula {i + 1}")
                
                plate_img = image[y:y+h, x:x+w]
                
                if plate_img.size == 0:
                    continue
                
                plate_img_large = cv2.resize(plate_img, None, fx=3, fy=3, 
                                              interpolation=cv2.INTER_CUBIC)
                
                segments = segment_plate_regions(plate_img_large)
                
                vis_plate = draw_plate_segmentation(plate_img_large, segments)
                vis_plate_rgb = cv2.cvtColor(vis_plate, cv2.COLOR_BGR2RGB)
                
                pcol1, pcol2 = st.columns(2)
                
                with pcol1:
                    st.markdown("**Matrícula extraída:**")
                    plate_rgb = cv2.cvtColor(plate_img_large, cv2.COLOR_BGR2RGB)
                    st.image(plate_rgb, use_container_width=True)
                
                with pcol2:
                    st.markdown("**Segmentación por zonas:**")
                    st.image(vis_plate_rgb, use_container_width=True)
                
                st.markdown("#### Componentes identificados:")
                
                comp_cols = st.columns(3)
                
                with comp_cols[0]:
                    st.markdown("🔵 **Banda de País**")
                    if segments['country_band'] is not None:
                        country_region = segments['country_band']['region']
                        country_rgb = cv2.cvtColor(country_region, cv2.COLOR_BGR2RGB)
                        st.image(country_rgb, width=80)
                        st.caption("Zona identificada")
                    else:
                        st.info("No detectada")
                
                with comp_cols[1]:
                    st.markdown("🔴 **Números**")
                    numbers = [c for c in segments['characters'] if c['type'] == 'number']
                    if numbers:
                        number_cols = st.columns(min(len(numbers), 4))
                        for j, number in enumerate(numbers[:4]):
                            with number_cols[j]:
                                number_rgb = cv2.cvtColor(number['region'], cv2.COLOR_BGR2RGB)
                                st.image(number_rgb, width=40)
                        st.caption(f"{len(numbers)} número(s) identificado(s)")
                    else:
                        st.info("No detectados")
                
                with comp_cols[2]:
                    st.markdown("🟢 **Letras**")
                    letters = [c for c in segments['characters'] if c['type'] == 'letter']
                    if letters:
                        letter_cols = st.columns(min(len(letters), 4))
                        for j, letter in enumerate(letters[:4]):
                            with letter_cols[j]:
                                letter_rgb = cv2.cvtColor(letter['region'], cv2.COLOR_BGR2RGB)
                                st.image(letter_rgb, width=40)
                        st.caption(f"{len(letters)} letra(s) identificada(s)")
                    else:
                        st.info("No detectadas")
                
                st.divider()
        else:
            with col2:
                st.warning("⚠️ No se detectaron matrículas en la imagen.")
                st.markdown("""
                **Consejos para mejorar la detección:**
                - Usa imágenes con buena iluminación
                - Asegúrate de que la matrícula sea visible y no esté muy inclinada
                - Prueba con imágenes donde la matrícula ocupe una parte significativa
                """)
    else:
        st.info("👆 Sube una imagen para comenzar el análisis")
        
        st.markdown("### ¿Cómo funciona?")
        st.markdown("""
        1. **Carga una imagen** con uno o más coches
        2. El sistema **detecta las matrículas** automáticamente
        3. Cada matrícula se **segmenta** en sus componentes:
           - Banda azul de país (típica en matrículas europeas)
           - Caracteres alfabéticos
           - Números
        4. Los componentes se **muestran separados** para su análisis
        """)


if __name__ == "__main__":
    main()
