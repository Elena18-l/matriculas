import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


def preprocess_image(image):
    """Convert image to grayscale and apply CLAHE for better contrast in low-light."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    
    return gray, enhanced


def detect_license_plates_single_pass(enhanced, img_shape, canny_low=30, canny_high=150):
    """Single detection pass with given parameters."""
    edges = cv2.Canny(enhanced, canny_low, canny_high)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    img_height, img_width = img_shape[:2]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = w * h
        img_area = img_width * img_height
        area_ratio = area / img_area
        
        if 1.5 < aspect_ratio < 7.0 and 0.002 < area_ratio < 0.25:
            if w > 40 and h > 10:
                plates.append((x, y, w, h))
    
    return plates


def detect_plate_in_cropped_image(image):
    """Detect if the whole image is essentially a license plate (cropped plate image)."""
    h, w = image.shape[:2]
    aspect_ratio = w / float(h) if h > 0 else 0
    
    if 1.5 < aspect_ratio < 7.0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_elements = 0
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            char_aspect = cw / float(ch) if ch > 0 else 0
            height_ratio = ch / float(h)
            width_ratio = cw / float(w)
            
            if ch > 10 and cw > 5:
                if (0.1 < char_aspect < 2.0 and 0.15 < height_ratio < 0.9) or \
                   (0.3 < width_ratio < 0.95 and 0.2 < height_ratio < 0.8):
                    valid_elements += 1
        
        if valid_elements >= 1:
            margin_x = int(w * 0.02)
            margin_y = int(h * 0.05)
            return [(margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)]
    
    return []


def detect_license_plates(image):
    """Detect potential license plate regions using multiple preprocessing techniques."""
    gray, enhanced = preprocess_image(image)
    img_shape = image.shape
    img_h, img_w = img_shape[:2]
    
    all_plates = []
    
    plates1 = detect_license_plates_single_pass(enhanced, img_shape, 30, 150)
    all_plates.extend(plates1)
    
    plates2 = detect_license_plates_single_pass(enhanced, img_shape, 50, 200)
    all_plates.extend(plates2)
    
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    plates3 = detect_license_plates_single_pass(adaptive, img_shape, 50, 150)
    all_plates.extend(plates3)
    
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plates4 = detect_license_plates_single_pass(otsu, img_shape, 50, 150)
    all_plates.extend(plates4)
    
    merged_plates = merge_overlapping_plates(all_plates)
    
    if not merged_plates:
        cropped_plates = detect_plate_in_cropped_image(image)
        if cropped_plates:
            return cropped_plates
    
    merged_plates = sorted(merged_plates, key=lambda p: p[2] * p[3], reverse=True)
    
    return merged_plates[:5]


def merge_overlapping_plates(plates):
    """Merge overlapping plate detections."""
    if not plates:
        return []
    
    plates = list(set(plates))
    
    merged = []
    used = [False] * len(plates)
    
    for i, (x1, y1, w1, h1) in enumerate(plates):
        if used[i]:
            continue
        
        current_x, current_y = x1, y1
        current_w, current_h = w1, h1
        
        for j, (x2, y2, w2, h2) in enumerate(plates):
            if i == j or used[j]:
                continue
            
            overlap_x = max(0, min(current_x + current_w, x2 + w2) - max(current_x, x2))
            overlap_y = max(0, min(current_y + current_h, y2 + h2) - max(current_y, y2))
            overlap_area = overlap_x * overlap_y
            
            area1 = current_w * current_h
            area2 = w2 * h2
            min_area = min(area1, area2)
            
            if overlap_area > 0.3 * min_area:
                new_x = min(current_x, x2)
                new_y = min(current_y, y2)
                new_w = max(current_x + current_w, x2 + w2) - new_x
                new_h = max(current_y + current_h, y2 + h2) - new_y
                current_x, current_y = new_x, new_y
                current_w, current_h = new_w, new_h
                used[j] = True
        
        merged.append((current_x, current_y, current_w, current_h))
        used[i] = True
    
    return merged


def get_position_based_segments(h, w, offset_x=0):
    """Get character segments based on Spanish plate format positions."""
    # Spanish plate: [country ~12%] [4 numbers + 3 letters in remaining 88%]
    country_width = int(w * 0.12)
    char_zone_start = country_width
    char_zone_width = w - country_width
    
    # 7 characters with some spacing
    num_chars = 7
    char_width = char_zone_width / num_chars
    
    # Character height is typically 40-70% of plate height, centered
    char_height = int(h * 0.55)
    char_y = int(h * 0.22)
    
    chars = []
    for i in range(num_chars):
        x_start = int(char_zone_start + i * char_width + char_width * 0.12)
        x_end = int(char_zone_start + (i + 1) * char_width - char_width * 0.12)
        
        chars.append((x_start + offset_x, char_y, x_end - x_start, char_height))
    
    return chars


def find_characters_in_image(gray_img, h, w, offset_x=0):
    """Try multiple methods to find characters, with position-based fallback."""
    
    # Method 1: Adaptive threshold with CLAHE
    denoised = cv2.GaussianBlur(gray_img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Try adaptive threshold
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 29, 5)
    
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / float(ch) if ch > 0 else 0
        height_ratio = ch / float(h) if h > 0 else 0
        
        if 0.15 < aspect_ratio < 1.2 and 0.2 < height_ratio < 0.85 and cw > 8:
            candidates.append((x + offset_x, y, cw, ch))
    
    # Method 2: Otsu threshold if adaptive didn't find enough
    if len(candidates) < 5:
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
        
        contours2, _ = cv2.findContours(otsu_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours2:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / float(ch) if ch > 0 else 0
            height_ratio = ch / float(h) if h > 0 else 0
            
            if 0.15 < aspect_ratio < 1.2 and 0.2 < height_ratio < 0.85 and cw > 8:
                candidates.append((x + offset_x, y, cw, ch))
    
    # Remove duplicates
    unique_chars = []
    for c in sorted(candidates, key=lambda x: x[2] * x[3], reverse=True):
        cx, cy, cw, ch = c
        is_overlap = False
        for uc in unique_chars:
            ux, uy, uw, uh = uc
            overlap_x = max(0, min(cx + cw, ux + uw) - max(cx, ux))
            if overlap_x > min(cw, uw) * 0.3:
                is_overlap = True
                break
        if not is_overlap:
            unique_chars.append(c)
    
    sorted_chars = sorted(unique_chars, key=lambda c: c[0])
    
    # Fallback: Use position-based segmentation if we didn't find enough characters
    if len(sorted_chars) < 5:
        sorted_chars = get_position_based_segments(h, w, offset_x)
    
    return sorted_chars


def segment_plate_regions(plate_img):
    """Segment the plate into country band, letters, and numbers regions."""
    h, w = plate_img.shape[:2]
    
    segments = {
        'country_band': None,
        'characters': [],
        'visualization': None
    }
    
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    country_band_width = int(w * 0.12)
    if country_band_width > 5:
        segments['country_band'] = {
            'region': plate_img[:, :country_band_width],
            'bbox': (0, 0, country_band_width, h),
            'type': 'country'
        }
    
    main_gray = gray[:, country_band_width:] if country_band_width > 5 else gray
    main_w = main_gray.shape[1]
    
    char_contours = find_characters_in_image(main_gray, h, main_w, country_band_width)
    
    for i, (x, y, cw, ch) in enumerate(char_contours):
        if y + ch > h or x + cw > w:
            continue
        char_region = plate_img[y:y+ch, x:x+cw]
        
        if char_region.size == 0:
            continue
        
        relative_pos = (x - country_band_width) / main_w if main_w > 0 else 0
        
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
