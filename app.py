import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology
import streamlit as st


# ---------- Utility functions ----------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_image(gray: np.ndarray) -> np.ndarray:
    """Normalize contrast to 0-255 uint8."""
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype("uint8")


def background_correction(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """
    Estimate background using a large median blur and subtract.
    kernel_size should be odd.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    bg = cv2.medianBlur(gray, kernel_size)
    corrected = cv2.subtract(gray, bg)
    return normalize_image(corrected)


def segment_colonies(
    gray: np.ndarray,
    min_size: int = 50,
    split_clusters: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment colonies on an agar plate.
    Returns:
        labels: labeled mask
        overlay_mask: binary mask of colonies
    """
    # Invert if colonies are dark on light background
    if gray.mean() > 127:
        inv = 255 - gray
    else:
        inv = gray

    # Threshold using Otsu
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small noise
    thresh = morphology.remove_small_objects(thresh.astype(bool), min_size=min_size)
    thresh = (thresh * 255).astype("uint8")

    if split_clusters:
        # Distance transform
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        # Find peaks
        _, sure_fg = cv2.threshold(dist_norm, 0.4, 1.0, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        # Connected components on peaks
        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # background not 0
        # Watershed
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color, markers)
        labels = markers.copy()
        labels[labels == -1] = 0  # boundaries -> background
        overlay_mask = (labels > 0).astype("uint8") * 255
    else:
        # Simple connected components
        labels = measure.label(thresh > 0, connectivity=2)
        overlay_mask = thresh

    return labels, overlay_mask


def compute_colony_stats(
    labels: np.ndarray, image_name: str
) -> Tuple[pd.DataFrame, int]:
    """
    Compute colony centroids, size (area in pixels).
    """
    props = measure.regionprops(labels)
    rows = []
    for i, region in enumerate(props, start=1):
        y, x = region.centroid  # (row, col) -> (y, x)
        rows.append(
            {
                "image_name": image_name,
                "object_id": i,
                "x": float(x),
                "y": float(y),
                "size": int(region.area),
            }
        )
    df = pd.DataFrame(rows)
    total = len(rows)
    if total > 0:
        df["total"] = total
    else:
        df["total"] = []
    return df, total


def colony_overlay_image(
    image_bgr: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """
    Draw contours around each colony on the original image.
    """
    overlay = image_bgr.copy()
    mask = (labels > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def segment_confluency(
    gray: np.ndarray,
    method: str = "Otsu",
) -> np.ndarray:
    """
    Segment cell-covered area in brightfield image.
    Returns binary mask (uint8 0/255).
    """
    # Cells often appear darker; invert if necessary
    if gray.mean() > 127:
        inv = 255 - gray
    else:
        inv = gray

    if method == "Otsu":
        _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Triangle":
        _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        # Fallback: adaptive threshold
        mask = cv2.adaptiveThreshold(
            inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2
        )

    # Clean up
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=100)
    mask = morphology.remove_small_holes(mask, area_threshold=100)
    mask = (mask * 255).astype("uint8")
    return mask


def compute_confluency(mask: np.ndarray, image_name: str) -> pd.DataFrame:
    covered = np.count_nonzero(mask)
    total_pixels = mask.size
    percent = 100.0 * covered / float(total_pixels) if total_pixels > 0 else 0.0
    df = pd.DataFrame(
        [
            {
                "image_name": image_name,
                "percent_confluence": percent,
            }
        ]
    )
    return df


def confluency_overlay_image(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay semi-transparent mask on image.
    """
    overlay = image_bgr.copy()
    color_mask = np.zeros_like(image_bgr)
    color_mask[:, :] = (0, 255, 0)
    alpha = 0.4
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        image_bgr[mask_bool], 1 - alpha, color_mask[mask_bool], alpha, 0
    )
    return overlay


# ---------- Streamlit UI ----------


def run_cfu_mode():
    st.header("Scenario A — CFU: Count in a Click")

    uploaded_files = st.file_uploader(
        "Upload one or more agar plate images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        min_size = st.slider("Min colony size (pixels)", 10, 2000, 80, 10)
    with col2:
        split_clusters = st.checkbox(
            "Split merged colonies (watershed-style)", value=True
        )

    if uploaded_files:
        if len(uploaded_files) == 1:
            st.info("I see a clear plate image — processing colonies now.")
        else:
            st.info(f"I see {len(uploaded_files)} plates — running batch analysis now.")

        all_rows: List[pd.DataFrame] = []
        preview_overlays = []

        for f in uploaded_files:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = f.name

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = normalize_image(gray)
            gray_corr = background_correction(gray)

            labels, overlay_mask = segment_colonies(
                gray_corr, min_size=min_size, split_clusters=split_clusters
            )
            df, total = compute_colony_stats(labels, image_name)
            all_rows.append(df)

            overlay_img = colony_overlay_image(image_bgr, labels)
            preview_overlays.append((image_name, overlay_img))

            # Save outputs
            base = Path(image_name).stem
            csv_path = OUTPUT_DIR / f"{base}_CFU_counts.csv"
            overlay_path = OUTPUT_DIR / f"{base}_CFU_overlay.png"
            mask_path = OUTPUT_DIR / f"{base}_CFU_mask.png"

            df.to_csv(csv_path, index=False)
            cv2.imwrite(str(overlay_path), overlay_img)
            cv2.imwrite(str(mask_path), overlay_mask)

            st.success(f"Count complete for {image_name}: {total} colonies.")
            st.caption(f"CSV and overlay saved to {csv_path} and {overlay_path}")

        # Show overlays
        for image_name, overlay_img in preview_overlays:
            st.subheader(f"Overlay preview — {image_name}")
            st.image(
                cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                caption=f"Detected colonies on {image_name}",
                use_column_width=True,
            )

        # Combined CSV for read-out slide
        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            combined_path = OUTPUT_DIR / "CFU_counts.csv"
            combined.to_csv(combined_path, index=False)
            st.info(f"Combined CFU counts saved to {combined_path}")


def run_confluency_mode():
    st.header("Scenario B — Confluency Estimation")

    uploaded_files = st.file_uploader(
        "Upload one or more brightfield images of cell monolayers",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="confluency_uploader",
    )

    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Threshold method",
            options=["Otsu", "Triangle", "Adaptive"],
            help="If confluency threshold wobbles, try switching methods.",
        )
    with col2:
        st.caption(
            "Tip: If the mask looks noisy, adjust illumination in your source image for the next run."
        )

    if uploaded_files:
        if len(uploaded_files) == 1:
            st.info("I see a monolayer field-of-view — estimating confluency now.")
        else:
            st.info(
                f"I see {len(uploaded_files)} fields — running batch confluency estimation now."
            )

        all_rows: List[pd.DataFrame] = []
        for f in uploaded_files:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = f.name

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = normalize_image(gray)
            gray_corr = background_correction(gray, kernel_size=31)

            method_key = method if method != "Adaptive" else "Adaptive"
            mask = segment_confluency(gray_corr, method=method_key)
            df = compute_confluency(mask, image_name)
            all_rows.append(df)

            percent = df["percent_confluence"].iloc[0]

            overlay_img = confluency_overlay_image(image_bgr, mask)

            base = Path(image_name).stem
            csv_path = OUTPUT_DIR / f"{base}_confluency.csv"
            overlay_path = OUTPUT_DIR / f"{base}_confluency_overlay.png"
            mask_path = OUTPUT_DIR / f"{base}_confluency_mask.png"

            df.to_csv(csv_path, index=False)
            cv2.imwrite(str(overlay_path), overlay_img)
            cv2.imwrite(str(mask_path), mask)

            st.success(
                f"Confluency estimated for {image_name}: {percent:.1f}% coverage."
            )
            st.caption(f"CSV and mask saved to {csv_path} and {overlay_path}")

            st.subheader(f"Mask preview — {image_name}")
            st.image(
                cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                caption=f"Estimated confluency: {percent:.1f}%",
                use_column_width=True,
            )

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            combined_path = OUTPUT_DIR / "confluency_results.csv"
            combined.to_csv(combined_path, index=False)
            st.info(f"Combined confluency results saved to {combined_path}")


def main():
    st.title("Count in a Click — CFU & Confluency Demo")
    st.write(
        "Quick local prototype to count colonies and estimate confluency, with CSV + overlay outputs."
    )

    mode = st.sidebar.radio(
        "Choose scenario",
        ["CFU: Colony Count", "Confluency: % Area Covered"],
    )

    st.sidebar.markdown("### Output folder")
    st.sidebar.code(str(OUTPUT_DIR.resolve()))

    if mode == "CFU: Colony Count":
        run_cfu_mode()
    else:
        run_confluency_mode()


if __name__ == "__main__":
    main()
