import base64
import os
import cv2
import numpy as np
import streamlit.components.v1 as components
from processing import ensure_bgr

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draggable_cropper")
_draggable_crop_raw = components.declare_component("draggable_cropper", path=_COMPONENT_DIR)


def draggable_crop_box(img_bgr, initial_points=None, key=None):
    """
    Renders an interactive canvas with 4 draggable corner dots directly on the image.

    Parameters:
        img_bgr: OpenCV BGR image (numpy array, 2D or 3D).
        initial_points: Optional 4x2 array or list of normalized coordinates [[x,y],...] in range [0..1].
        key: Streamlit component key string.

    Returns:
        np.ndarray of shape (4, 2) in absolute pixel coordinates [[x, y], ...] on original image.
    """
    img_bgr = ensure_bgr(img_bgr)
    h, w = img_bgr.shape[:2]

    # Downscale for ultra-smooth 60fps browser rendering
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        preview = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview = img_bgr

    _, buffer = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    if initial_points is None:
        init_norm = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
    else:
        init_norm = [[float(p[0]), float(p[1])] for p in initial_points]

    res_norm = _draggable_crop_raw(
        image_base64=b64_str,
        initial_points=init_norm,
        max_width=640,
        default=init_norm,
        key=key,
    )

    if res_norm is None or len(res_norm) != 4:
        res_norm = init_norm

    # Convert normalized [0..1] coordinates back to original full-resolution pixel space
    pixel_pts = np.float32([
        [res_norm[0][0] * w, res_norm[0][1] * h],
        [res_norm[1][0] * w, res_norm[1][1] * h],
        [res_norm[2][0] * w, res_norm[2][1] * h],
        [res_norm[3][0] * w, res_norm[3][1] * h],
    ])
    return pixel_pts
