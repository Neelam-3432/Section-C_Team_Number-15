import cv2
import numpy as np


def ensure_bgr(img):
    """
    Ensures that the input image is a valid 3-channel uint8 BGR numpy array.
    Guarantees channel invariance across all IPCV processing tools in the pipeline.
    Handles:
    - 2D grayscale / binary images (H, W) -> converted to 3-channel BGR.
    - 3D 1-channel images (H, W, 1) -> squeezed and converted to 3-channel BGR.
    - 4-channel BGRA images (H, W, 4) -> converted to 3-channel BGR.
    - 3-channel BGR images (H, W, 3) -> returned as valid uint8 array.
    """
    if img is None:
        return None

    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif channels == 3:
            return img
        elif channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported image shape for BGR conversion: {img.shape}")


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
        Filtered image as standard 3-channel BGR numpy array.
    """
    img_bgr = ensure_bgr(img)
    if method == "none" or method is None:
        return img_bgr.copy()

    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1

    if method == "gaussian":
        res = cv2.GaussianBlur(img_bgr, (k, k), 0)
    elif method == "median":
        res = cv2.medianBlur(img_bgr, k)
    else:
        raise ValueError(f"Unknown filter method '{method}'. Supported: 'gaussian', 'median', 'none'")
    return ensure_bgr(res)


def is_photo(img):
    """
    Determines if the image is a colorful photo or a text document.
    Text documents have predominantly neutral white/gray/black tones (low saturation),
    whereas photos contain diverse chromatic color distributions.
    """
    if img is None:
        return False
    # Grayscale has zero chroma -> text page
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return False

    img_bgr = ensure_bgr(img)
    small = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # S = saturation (color intensity), V = brightness
    colourful = (hsv[..., 1] > 60) & (hsv[..., 2] > 50)
    return float(colourful.mean()) > 0.20


def auto_crop(img):
    """
    Trims outer margins around text content.
    Uses Otsu thresholding and morphological rectangular dilation to identify text bounding area.
    """
    if img is None:
        return img
    img_bgr = ensure_bgr(img)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(15, w // 12)
    blocks = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    boxes = [cv2.boundingRect(c) for c in contours]
    biggest_area = max(bw * bh for _, _, bw, bh in boxes)
    keep = [b for b in boxes if b[2] * b[3] >= 0.10 * biggest_area]
    if not keep:
        return img_bgr

    x1 = min(b[0] for b in keep)
    y1 = min(b[1] for b in keep)
    x2 = max(b[0] + b[2] for b in keep)
    y2 = max(b[1] + b[3] for b in keep)

    if (x2 - x1) * (y2 - y1) < 0.10 * w * h:
        return img_bgr
    return img_bgr[y1:y2, x1:x2]


def order_points(pts):
    """
    Topologically orders 4 arbitrary quadrilateral points:
    [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
    Using Euclidean coordinate sum (x + y) and difference (y - x).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Smallest sum = Top-Left
    rect[3] = pts[np.argmax(s)]  # Largest sum = Bottom-Right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Smallest difference = Top-Right
    rect[2] = pts[np.argmax(diff)]  # Largest difference = Bottom-Left
    return rect


def crop_and_straighten(img, pts):
    """
    Rectifies perspective distortion from 4 user-selected corner points using 2D homography.

    IPCV Syllabus Mapping - Unit I:
    Computes a 3x3 projective transformation matrix H via cv2.getPerspectiveTransform
    and remaps coordinates via bilinear interpolation (cv2.warpPerspective)
    into a front-facing, upright rectangular image.

    Parameters:
        img: Input image (numpy array, 2D or 3D BGR).
        pts: np.ndarray of shape (4, 2) in order [TL, TR, BL, BR].

    Returns:
        Rectified, straightened 3-channel BGR image.
    """
    if img is None:
        return None
    img_bgr = ensure_bgr(img)
    rect = order_points(pts)
    tl, tr, bl, br = rect

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 30 or height < 30:
        return img_bgr

    pts1 = np.float32(rect)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img_bgr, matrix, (width, height))
    return ensure_bgr(warped)


def rotate_image(img, angle=0):
    """
    Rotates an image by a specified angle in degrees.

    IPCV Syllabus Mapping - Unit I:
    Geometric Transformations (Affine 2D Rotation Matrix).
    - 90, 180, 270 degrees: lossless orthogonal matrix rotation.
    - Arbitrary angles: calculates 2x3 affine rotation matrix around image center
      and resamples via cv2.warpAffine.
    """
    if img is None:
        return None
    img_bgr = ensure_bgr(img)

    if angle == 0 or angle is None:
        return img_bgr.copy()

    # Orthogonal shortcuts
    if angle in (90, -270):
        res = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle in (180, -180):
        res = cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif angle in (270, -90):
        res = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Arbitrary angle rotation via 2x3 Affine Rotation Matrix (Unit I)
        h, w = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        res = cv2.warpAffine(img_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return ensure_bgr(res)


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

    Returns:
        Standard 3-channel BGR binary image (values 0 and 255) for pipeline invariance.
    """
    img_bgr = ensure_bgr(img)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bsize = max(3, int(block_size))
    if bsize % 2 == 0:
        bsize += 1

    if method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_gaussian":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, c
        )
    elif method == "adaptive_mean":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bsize, c
        )
    else:
        raise ValueError(
            f"Unknown threshold method '{method}'. Supported: 'otsu', 'adaptive_gaussian', 'adaptive_mean'"
        )

    return ensure_bgr(thresh)


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
    img_bgr = ensure_bgr(img)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
        Equalized standard 3-channel BGR image.
    """
    img_bgr = ensure_bgr(img)

    if method == "global":
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        res = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        res = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        raise ValueError(f"Unknown equalization method '{method}'. Supported: 'global', 'clahe'")

    return ensure_bgr(res)


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


def enhance_text_page(img):
    """
    Illumination compensation and text contrast enhancement:
    1. Background illumination modeling via morphological dilation & median filtering.
    2. Background division to flatten uneven shadows and gradients.
    3. Contrast stretching via non-linear Look-Up Table (LUT).
    4. Unsharp masking for crisp letter definition.
    """
    img_bgr = ensure_bgr(img)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

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
    sharp = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    return ensure_bgr(sharp)


def enhance_photo(img):
    """
    Photo enhancement in CIE-LAB color space:
    1. Extracts luminance channel L while preserving chromaticity channels A and B.
    2. Applies CLAHE to L channel to balance dynamic range across localized contextual tiles.
    3. Reconstructs BGR and sharpens using unsharp masking.
    """
    img_bgr = ensure_bgr(img)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    sharp = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    return ensure_bgr(sharp)


def enhance_photo_gray(img):
    """
    Monochrome photo enhancement:
    1. Grayscale luminance conversion.
    2. Adaptive contrast equalization via CLAHE.
    3. High-frequency edge emphasis via unsharp masking.
    """
    img_bgr = ensure_bgr(img)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)
    return ensure_bgr(sharp)


def process_image(img, mode="auto", output="bw"):
    """
    Full document enhancement pipeline:
    Automatically balances illumination, enhances text readability, or applies CLAHE
    based on picture type and desired output format.
    """
    img = ensure_bgr(img)
    h, w = img.shape[:2]
    scale = 2600.0 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if mode == "photo":
        photo = True
    elif mode == "text":
        photo = False
    else:
        photo = is_photo(img)

    if not photo:
        res = enhance_text_page(img)
    elif output == "bw":
        res = enhance_photo_gray(img)
    else:
        res = enhance_photo(img)

    return ensure_bgr(res)
