

import cv2
import numpy as np


def apply_noise_filter(img, method="gaussian", ksize=5):
    """
    Applies noise reduction filter to an image (Color or Grayscale).

    IPCV Syllabus Mapping - Unit II:
    - Gaussian Filter: Linear spatial smoothing filter. Convolves the image with
      a 2D Gaussian kernel. Effective for attenuating high-frequency Gaussian noise.
    - Median Filter: Non-linear order-statistic filter. Replaces each pixel with
      the median intensity in its neighborhood. Preserves sharp document edges while
      eliminating salt-and-pepper / impulse noise.

    Parameters:
        img: Input image (numpy array, 2D grayscale or 3D BGR).
        method: "gaussian", "median", or "none".
        ksize: Aperture linear size. Must be an odd integer >= 3 (e.g. 3, 5, 7).

    Returns:
        Filtered image as numpy array.
    """
    if method == "none" or method is None:
        return img.copy()

    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1

    if method == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    elif method == "median":
        return cv2.medianBlur(img, k)
    else:
        raise ValueError(f"Unknown filter method '{method}'. Supported: 'gaussian', 'median', 'none'")


def is_photo(img):
    """Guess if the picture is a PHOTO (face, scenery) or a TEXT PAGE.
    A text page is almost only white paper and black ink, so it has
    almost NO colour. A photo has a lot of colour (blue wall, skin...).
    We count how many pixels are 'colourful'."""
    small = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # S = saturation (how strong the colour is), V = brightness
    colourful = (hsv[..., 1] > 60) & (hsv[..., 2] > 50)
    return float(colourful.mean()) > 0.20


def auto_crop(img):
    """Cuts the empty borders around a TEXT page. Ignores tiny specks."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(15, w // 12)
    blocks = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    boxes = [cv2.boundingRect(c) for c in contours]
    biggest_area = max(bw * bh for _, _, bw, bh in boxes)
    keep = [b for b in boxes if b[2] * b[3] >= 0.10 * biggest_area]
    x1 = min(b[0] for b in keep)
    y1 = min(b[1] for b in keep)
    x2 = max(b[0] + b[2] for b in keep)
    y2 = max(b[1] + b[3] for b in keep)

    if (x2 - x1) * (y2 - y1) < 0.10 * w * h:
        return img
    return img[y1:y2, x1:x2]


def order_points(pts):
    """Orders points: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    return rect


def warp_perspective(img, biggest, crop=True):
    """Flattens the paper into a straight rectangle."""
    if biggest is None:
        return img

    rect = order_points(biggest)
    tl, tr, bl, br = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return img

    pts1 = np.float32(rect)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (width, height))
    return auto_crop(img_warp) if crop else img_warp


def get_contours(edges, min_area=500):
    """Finds the biggest 4-corner shape. We assume it is the paper."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, max_area = None, 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                biggest, max_area = approx, area

    return biggest.reshape(4, 2) if biggest is not None else None


def rotate_image(img, angle=0):
    """
    Rotates an image by a specified angle in degrees.

    IPCV Syllabus Mapping - Unit I:
    Geometric Transformations (Affine 2D Rotation Matrix).
    - 90, 180, 270 degrees: lossless orthogonal matrix rotation.
    - Arbitrary angles: calculates 2x3 affine rotation matrix around image center
      and resamples via cv2.warpAffine.
    """
    if angle == 0 or angle is None:
        return img.copy()

    # Right-angle shortcuts
    if angle in (90, -270):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle in (180, -180):
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle in (270, -90):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Arbitrary angle rotation via 2x3 Affine Rotation Matrix (Unit I)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def detect_document_corners(img):
    """
    Detects the 4 corners of the document boundary.

    IPCV Syllabus Mapping - Unit I & III:
    Uses Gaussian smoothing, Canny edge detection (50, 150), morphological closing,
    and polygon approximation (approxPolyDP) to segment the document paper quadrilateral boundary.

    Returns:
        numpy.ndarray of shape (4, 2) in original image coordinate space, or None.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (960, 1280))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else small.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Standard Canny gradient with morphological close
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.05 * 960 * 1280

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area > min_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32) * np.array([w / 960.0, h / 1280.0], dtype=np.float32)

    # 2. Adaptive Otsu fallback for soft indoor lighting
    _, otsu_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_otsu = cv2.Canny(otsu_bin, 50, 150)
    edges_otsu = cv2.morphologyEx(edges_otsu, cv2.MORPH_CLOSE, kernel)
    contours_otsu, _ = cv2.findContours(edges_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in sorted(contours_otsu, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area > min_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32) * np.array([w / 960.0, h / 1280.0], dtype=np.float32)

    return None


def visualize_document_boundary(img, corners=None):
    """
    Draws document boundary polygon and highlighted corner markers (TL, TR, BL, BR).

    IPCV Syllabus Mapping - Unit I & III:
    Visualizes pixel coordinates, spatial orientation, and quadrilateral segmentation.
    """
    annotated = img.copy()
    if corners is None:
        corners = detect_document_corners(img)

    if corners is not None:
        ordered = order_points(corners)
        pts = ordered.astype(np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
        labels = ["TL", "TR", "BL", "BR"]
        colors = [(255, 0, 0), (0, 165, 255), (0, 0, 255), (255, 0, 255)]  # BGR
        radius = max(6, int(min(annotated.shape[:2]) * 0.015))
        for (x, y), label, color in zip(pts, labels, colors):
            cv2.circle(annotated, (int(x), int(y)), radius, color, -1)
            cv2.putText(
                annotated,
                label,
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
    return annotated


def apply_threshold(img, method="otsu", block_size=11, c=2):
    """
    Applies global or adaptive thresholding to convert image to binary representation.

    IPCV Syllabus Mapping - Unit III:
    - Otsu Thresholding: Global optimal thresholding. Automatically calculates optimum
      threshold t that maximizes between-class variance sigma_B^2(t).
    - Adaptive Gaussian Thresholding: Local threshold calculated as weighted Gaussian
      sum of neighboring window (block_size x block_size) minus constant C.
    - Adaptive Mean Thresholding: Local threshold calculated as arithmetic mean of
      neighboring window minus constant C.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    bsize = max(3, int(block_size))
    if bsize % 2 == 0:
        bsize += 1

    if method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    elif method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c
        )
    elif method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bsize, c
        )
    else:
        raise ValueError(
            f"Unknown threshold method '{method}'. Supported: 'otsu', 'adaptive_gaussian', 'adaptive_mean'"
        )


def compute_grayscale_histogram(img):
    """
    Computes 256-bin intensity histogram of a grayscale or BGR image.

    IPCV Syllabus Mapping - Unit II:
    Represents the discrete probability distribution of pixel intensity levels r_k.
    p(r_k) = n_k / MN, where n_k is number of pixels with gray level r_k.

    Parameters:
        img: 2D grayscale array or 3D BGR array.

    Returns:
        1D numpy array of length 256 containing raw pixel counts.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten()


def apply_histogram_equalization(img, method="global", clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Performs Global Histogram Equalization or CLAHE.

    IPCV Syllabus Mapping - Unit II:
    - Global Equalization: Spreads out frequent intensity values using cumulative distribution
      function (CDF) mapping s_k = T(r_k) = (L - 1) * sum_{j=0}^k (n_j / MN).
    - CLAHE (Contrast Limited Adaptive Histogram Equalization): Divides image into contextual
      tiles (e.g. 8x8), clips histogram peaks to avoid noise amplification, and interpolates
      tile boundaries using bilinear interpolation.

    Parameters:
        img: Input image (2D grayscale or 3D BGR).
        method: "global" or "clahe".
        clip_limit: Contrast threshold for CLAHE.
        tile_grid_size: Tile grid tuple for CLAHE.

    Returns:
        Equalized image matching input format.
    """
    is_color = img.ndim == 3

    if method == "global":
        if is_color:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(img)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if is_color:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return clahe.apply(img)
    else:
        raise ValueError(f"Unknown equalization method '{method}'. Supported: 'global', 'clahe'")


def render_histogram_image(img, width=256, height=120, color=(52, 152, 219)):
    """
    Renders a 256-bin intensity histogram as a visual BGR image.
    Uses pure OpenCV and NumPy (zero external graphing/pandas dependencies).
    """
    hist = compute_grayscale_histogram(img)
    hist_img = np.full((height, width, 3), 245, dtype=np.uint8)
    max_val = float(hist.max()) if hist.max() > 0 else 1.0
    normalized = (hist / max_val) * (height - 12)
    for x in range(min(width, len(hist))):
        val = int(normalized[x])
        if val > 0:
            cv2.line(hist_img, (x, height - 1), (x, height - 1 - val), color, 1)
    return hist_img


def draw_contour(img, biggest):
    if biggest is not None:
        cv2.polylines(img, [biggest.astype(np.int32)], True, (0, 255, 0), 3)
    return img


def enhance_text_page(img):
    """For TEXT pages: even light, white paper, dark sharp text (grey output)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < 1600:
        s = 1600.0 / w
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape

    k1 = max(7, (w // 200) | 1)
    k2 = max(21, (w // 60) | 1)
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((k1, k1), np.uint8)), k2)
    norm = cv2.divide(gray, bg, scale=255)

    lut = np.clip((np.arange(256) - 70) * 255.0 / (225 - 70), 0, 255).astype(np.uint8)
    out = cv2.LUT(norm, lut)

    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def enhance_photo(img):
    """For PHOTOS: keep the colours and the faces. Only fix light a little
    and make it a bit sharper (colour output)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE = makes dark and bright parts balanced, in small tiles
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def enhance_photo_gray(img):
    """For PHOTOS in black and white, like a printer does.
    Step 1: change colour to shades of grey (grayscale).
    Step 2: balance the light, so face and shirt are both visible.
    Step 3: sharpen a little."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # colour -> grey shades

    # CLAHE = balances dark and bright parts in small tiles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.addWeighted(gray, 1.4, blur, -0.4, 0)


def process_image(img, mode="auto", output="bw"):
    """Full pipeline: BGR image in -> scanned page out.
    mode   = "auto" (decide by itself), "text" or "photo"
    output = "bw" (black and white, like a printer) or "color" """
    h, w = img.shape[:2]
    scale = 2600.0 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    if mode == "photo":
        photo = True
    elif mode == "text":
        photo = False
    else:
        photo = is_photo(img)

    # 1) find the paper corners on a small copy
    small = cv2.resize(img, (960, 1280))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 190, 190)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    biggest = detect_document_corners(img)

    # 2) flatten (photos are NOT cropped by ink, so we never cut the face)
    if biggest is not None:
        biggest = biggest.astype(np.float32)
        warped = warp_perspective(img, biggest, crop=not photo)
    else:
        warped = img if photo else auto_crop(img)

    # 3) clean it up in the right way
    if not photo:
        return enhance_text_page(warped)          # text: always clean grey/white
    if output == "bw":
        return enhance_photo_gray(warped)         # photo in black and white
    return enhance_photo(warped)                  # photo in colour