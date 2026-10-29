from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from cropper_component import draggable_crop_box
from processing import (
    apply_histogram_equalization,
    apply_noise_filter,
    apply_threshold,
    crop_and_straighten,
    detect_document_corners,
    draw_crop_box,
    ensure_bgr,
    order_points,
    process_image,
    rotate_image,
    straighten_document,
    visualize_document_boundary,
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
    img_bgr = ensure_bgr(img_array)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
        "Detect Corners & Straighten": "Capture and straighten any document using an interactive 4-dot crop box.",
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
    if selected_tool == "Detect Corners & Straighten":
        operation = "Detect Corners & Straighten"

    elif selected_tool == "Scan a Document":
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

    # Pipeline State Memory (Keep / Remove Result)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Keep / Remove Result")
    st.sidebar.caption("Lock in results so switching to another dropdown won't erase your progress.")
    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        if st.sidebar.button("💾 Keep All", key="sb_keep_all", help="Save current result for all pages as input for next dropdown tool"):
            uploaded_count = len(st.session_state.get("current_uploaded_files", []))
            for i in range(uploaded_count):
                if f"latest_res_{i}" in st.session_state:
                    if f"history_stack_{i}" not in st.session_state:
                        st.session_state[f"history_stack_{i}"] = []
                    st.session_state[f"history_stack_{i}"].append((operation, ensure_bgr(st.session_state[f"latest_res_{i}"].copy())))
            st.rerun()
    with col_sb2:
        if st.sidebar.button("🗑️ Reset All", key="sb_reset_all", help="Reset all pages back to original photo"):
            uploaded_count = len(st.session_state.get("current_uploaded_files", []))
            for i in range(uploaded_count):
                st.session_state[f"history_stack_{i}"] = []
            st.rerun()

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
        st.session_state["current_uploaded_files"] = uploaded_files
        scanned_images = []

        with st.spinner("🔍 Processing documents..."):
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file)
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    raw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # 1. Page Orientation / Rotation (Member 4 - Unit I Geometric Transformation)
                    if rotate_option == "90° Clockwise":
                        raw_img = rotate_image(raw_img, 90)
                    elif rotate_option == "180° Flip":
                        raw_img = rotate_image(raw_img, 180)
                    elif rotate_option == "270° Counter-Clockwise":
                        raw_img = rotate_image(raw_img, 270)

                    # Track file identity to reset history if new file uploaded
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                    if st.session_state.get(f"file_id_{idx}") != file_id:
                        st.session_state[f"file_id_{idx}"] = file_id
                        st.session_state[f"history_stack_{idx}"] = []

                    history_stack = st.session_state.get(f"history_stack_{idx}", [])

                    # Working input: use previous kept result if available, else rotated raw photo
                    if history_stack:
                        working_img = ensure_bgr(history_stack[-1][1].copy())
                        input_caption = f"Input Image (Kept: {history_stack[-1][0]})"
                    else:
                        working_img = ensure_bgr(raw_img.copy())
                        input_caption = "Original Image"

                    h, w = working_img.shape[:2]

                    st.markdown(f"#### 📄 Page {idx + 1}: {uploaded_file.name}")

                    # 2. Main Operation Execution
                    if operation == "Detect Corners & Straighten":
                        # Smart initial default positions
                        auto_c = detect_document_corners(working_img)
                        if auto_c is not None:
                            ord_pts = order_points(auto_c)
                            init_norm = [
                                [float(np.clip(ord_pts[0][0] / w, 0.01, 0.99)), float(np.clip(ord_pts[0][1] / h, 0.01, 0.99))],
                                [float(np.clip(ord_pts[1][0] / w, 0.01, 0.99)), float(np.clip(ord_pts[1][1] / h, 0.01, 0.99))],
                                [float(np.clip(ord_pts[2][0] / w, 0.01, 0.99)), float(np.clip(ord_pts[2][1] / h, 0.01, 0.99))],
                                [float(np.clip(ord_pts[3][0] / w, 0.01, 0.99)), float(np.clip(ord_pts[3][1] / h, 0.01, 0.99))],
                            ]
                        else:
                            init_norm = [[0.06, 0.06], [0.94, 0.06], [0.06, 0.94], [0.94, 0.94]]

                        c_left, c_right = st.columns([1.1, 0.9])
                        with c_left:
                            st.markdown("##### 🖐️ Interactive 4-Dot Canvas")
                            st.caption("Drag the 4 colored dots to position corners on the document:")
                            pts = draggable_crop_box(working_img, initial_points=init_norm, key=f"drag_crop_{idx}")

                        with c_right:
                            st.markdown("##### 📄 Straightened Scan")
                            result = crop_and_straighten(working_img, pts)
                            if output == "bw":
                                gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
                                result = cv2.cvtColor(gray_res, cv2.COLOR_GRAY2BGR)

                            scanned_pil = to_pil(result)
                            st.image(scanned_pil, caption="Straightened Document (Inside 4 Dots)", use_container_width=True)
                            st.success("✅ Perspective rectified from 4 draggable corner dots.")

                    else:
                        if operation == "Full Document Scanner":
                            result = process_image(working_img, mode=mode, output=output)
                        elif operation == "Gaussian Smoothing":
                            result = apply_noise_filter(working_img, method="gaussian", ksize=filter_ksize)
                        elif operation == "Median Filtering":
                            result = apply_noise_filter(working_img, method="median", ksize=filter_ksize)
                        elif operation == "Otsu Thresholding":
                            result = apply_threshold(working_img, method="otsu")
                        elif operation == "Adaptive Thresholding":
                            result = apply_threshold(working_img, method="adaptive_gaussian", block_size=11, c=2)
                        elif operation == "Global Histogram Equalization":
                            result = apply_histogram_equalization(working_img, method="global")
                        elif operation == "CLAHE (Adaptive Equalization)":
                            result = apply_histogram_equalization(working_img, method="clahe")
                        else:
                            result = process_image(working_img, mode=mode, output=output)

                        scanned_pil = to_pil(result)
                        c_left, c_right = st.columns(2)
                        with c_left:
                            st.image(to_pil(working_img), caption=input_caption, use_container_width=True)
                        with c_right:
                            st.image(scanned_pil, caption=f"Result: {operation}", use_container_width=True)

                    result = ensure_bgr(result)
                    scanned_images.append(scanned_pil)
                    st.session_state[f"latest_res_{idx}"] = result.copy()

                    # --- Action Bar: Keep & Remove / Revert Controls for this page ---
                    col_k1, col_k2, col_k3 = st.columns([1, 1, 1.8])
                    with col_k1:
                        if st.button("💾 Keep This Result", key=f"btn_keep_{idx}", help="Save this result so you can apply other dropdown tools on top of it without losing it"):
                            if f"history_stack_{idx}" not in st.session_state:
                                st.session_state[f"history_stack_{idx}"] = []
                            st.session_state[f"history_stack_{idx}"].append((operation, result.copy()))
                            st.rerun()

                    with col_k2:
                        has_history = len(st.session_state.get(f"history_stack_{idx}", [])) > 0
                        if st.button("🗑️ Remove / Revert", key=f"btn_revert_{idx}", disabled=not has_history, help="Discard kept steps and revert back to original photo"):
                            st.session_state[f"history_stack_{idx}"] = []
                            st.rerun()

                    with col_k3:
                        steps = [step[0] for step in st.session_state.get(f"history_stack_{idx}", [])]
                        if steps:
                            st.success(f"📌 Kept Chain: **{' ➔ '.join(steps)}**")
                        else:
                            st.caption("📌 Working on original image (click 'Keep' to save this step)")

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
