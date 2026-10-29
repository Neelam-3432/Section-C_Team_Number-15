from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from processing import (
    apply_histogram_equalization,
    apply_noise_filter,
    apply_threshold,
    process_image,
    rotate_image,
)

# Must be the FIRST Streamlit command
st.set_page_config(page_title="Scanify - Document Scanner", page_icon="📄", layout="wide")

CSS = """
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
    .header { text-align: center; padding: 1.5rem 0;
              border-bottom: 2px solid #e0e0e0; margin-bottom: 1.5rem; }
    .title { color: #2c3e50; font-size: 2.3rem; font-weight: 700; margin: 0.5rem 0; }
    .subheader { color: #7f8c8d; font-size: 1.05rem; }
    .stDownloadButton button { background: #3498db !important; color: white !important;
                               border-radius: 25px; padding: 0.5rem 2rem; transition: all 0.3s; }
    .stDownloadButton button:hover { background: #2980b9 !important; transform: translateY(-1px); }
</style>
"""

MODES = {
    "Auto (recommended)": "auto",
    "Text page": "text",
    "Photo (face / colour picture)": "photo",
}

OUTPUTS = {
    "Black & white (like a printer)": "bw",
    "Colour": "color",
}


def to_pil(img_array):
    """Turns a numpy image array into a PIL Image for display and PDF saving.
    Grayscale (2D) or BGR (3D) -> RGB PIL Image."""
    if img_array.ndim == 3:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="header">
            <h1 class="title">📄 Scanify</h1>
            <p class="subheader">IPCV Document Scanner & Image Enhancement Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar: Image Tools ---
    st.sidebar.markdown("## 🛠️ Image Tools")

    TOOL_DESCRIPTIONS = {
        "Scan a Document": "Detect document edges, straighten the page, and improve readability.",
        "Remove Image Noise": "Reduce graininess or speckles using Gaussian or Median filter.",
        "Make Text Stand Out": "Convert image to black & white using Otsu or Adaptive thresholding.",
        "Improve Image Clarity": "Enhance contrast using Global Equalization or CLAHE.",
    }

    selected_tool = st.sidebar.selectbox(
        "What would you like to do?",
        list(TOOL_DESCRIPTIONS.keys()),
        index=0,
        help="Choose a document processing tool.",
    )

    # Description box beneath dropdown for accessibility
    st.sidebar.info(f"💡 {TOOL_DESCRIPTIONS[selected_tool]}")

    filter_ksize = 5
    if selected_tool == "Scan a Document":
        operation = "Full Document Scanner"

    elif selected_tool == "Remove Image Noise":
        operation = st.sidebar.radio(
            "Filter Type",
            ["Gaussian Smoothing", "Median Filtering"],
            index=0,
            help="Gaussian is for general grain; Median is for salt-and-pepper noise.",
        )
        filter_ksize = st.sidebar.slider(
            "Filter Strength (Kernel Size)",
            min_value=3,
            max_value=11,
            value=5,
            step=2,
            help="Higher values apply stronger smoothing.",
        )

    elif selected_tool == "Make Text Stand Out":
        operation = st.sidebar.radio(
            "Thresholding Method",
            ["Otsu Thresholding", "Adaptive Thresholding"],
            index=0,
            help="Otsu is best for even light; Adaptive is best for uneven shadows.",
        )

    elif selected_tool == "Improve Image Clarity":
        operation = st.sidebar.radio(
            "Enhancement Method",
            ["Global Histogram Equalization", "CLAHE (Adaptive Equalization)"],
            index=0,
            help="Global spreads all shades; CLAHE balances contrast locally in small tiles.",
        )

    # Orientation / Rotation Controls (Member 4 - Unit I Geometric Transformations)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Orientation / Rotation")
    rotate_option = st.sidebar.selectbox(
        "Rotate Page",
        ["None (0°)", "90° Clockwise", "180° Flip", "270° Counter-Clockwise"],
        index=0,
        help="Rotate sideways mobile photos.",
    )

    # --- Main Page Settings ---
    col1, col2 = st.columns(2)
    with col1:
        mode_name = st.radio("Picture type", list(MODES.keys()), horizontal=True)
    with col2:
        out_name = st.radio("Output", list(OUTPUTS.keys()), horizontal=True)
    mode = MODES[mode_name]
    output = OUTPUTS[out_name]

    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload document images",
        key="multi_uploader",
    )

    if uploaded_files:
        scanned_images = []

        with st.spinner("🔍 Processing documents..."):
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file)
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # 1. Page Orientation / Rotation (Member 4 - Unit I Geometric Transformation)
                    if rotate_option == "90° Clockwise":
                        img = rotate_image(img, 90)
                    elif rotate_option == "180° Flip":
                        img = rotate_image(img, 180)
                    elif rotate_option == "270° Counter-Clockwise":
                        img = rotate_image(img, 270)

                    # 2. Main Operation Execution
                    if operation == "Full Document Scanner":
                        result = process_image(img, mode=mode, output=output)
                    elif operation == "Gaussian Smoothing":
                        result = apply_noise_filter(img, method="gaussian", ksize=filter_ksize)
                    elif operation == "Median Filtering":
                        result = apply_noise_filter(img, method="median", ksize=filter_ksize)
                    elif operation == "Otsu Thresholding":
                        result = apply_threshold(img, method="otsu")
                    elif operation == "Adaptive Thresholding":
                        result = apply_threshold(img, method="adaptive_gaussian", block_size=11, c=2)
                    elif operation == "Global Histogram Equalization":
                        result = apply_histogram_equalization(img, method="global")
                    elif operation == "CLAHE (Adaptive Equalization)":
                        result = apply_histogram_equalization(img, method="clahe")
                    else:
                        result = process_image(img, mode=mode, output=output)

                    scanned_pil = to_pil(result)
                    scanned_images.append(scanned_pil)

                    # 3. Side-by-Side Display
                    st.markdown(f"#### 📄 Page {idx + 1}: {uploaded_file.name}")
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.image(to_pil(img), caption="Original Image", use_container_width=True)
                    with c_right:
                        st.image(scanned_pil, caption=f"Result: {operation}", use_container_width=True)

                    st.markdown("---")

                except Exception as e:
                    st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")

        if scanned_images:
            st.markdown("### 📥 Download Your Scanned PDF")
            pdf_bytes = BytesIO()
            scanned_images[0].save(
                pdf_bytes,
                format="PDF",
                save_all=True,
                append_images=scanned_images[1:],
            )
            pdf_bytes.seek(0)

            st.download_button(
                label="Download Scanned PDF",
                data=pdf_bytes,
                file_name="scanned_document.pdf",
                mime="application/pdf",
            )

    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d; margin-top: 3rem;">
            <p>Scanify • IPCV Lab Project </p>
            <p>Powered by OpenCV & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
