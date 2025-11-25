import io
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology
import streamlit as st


# ---------- Utility functions ----------

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
        # Adaptive
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


# ---------- CFU mode (UI-focused) ----------

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
        summary_rows = []

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

            # ---- UI: per-image results ----
            st.subheader(f"Results — {image_name}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total colonies", total)
            with c2:
                if len(df) > 0:
                    st.metric("Median colony size (px)", int(df["size"].median()))
                else:
                    st.metric("Median colony size (px)", 0)

            st.caption("Per-colony measurements")
            if len(df) > 0:
                st.dataframe(
                    df[["object_id", "x", "y", "size"]],
                    use_container_width=True,
                )
            else:
                st.write("No colonies detected with current parameters.")

            # Overlay preview
            overlay_img = colony_overlay_image(image_bgr, labels)
            st.image(
                cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                caption=f"Detected colonies on {image_name}",
                use_column_width=True,
            )

            # Optional: downloadable CSV for this image
            if len(df) > 0:
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button(
                    label=f"Download CSV for {image_name}",
                    data=csv_buf.getvalue(),
                    file_name=f"{Path(image_name).stem}_CFU_counts.csv",
                    mime="text/csv",
                )

            # Save a small summary row for cross-image table
            summary_rows.append(
                {
                    "image_name": image_name,
                    "total_colonies": total,
                    "median_size_px": int(df["size"].median()) if len(df) > 0 else 0,
                }
            )

        # ---- UI: summary across images ----
        if summary_rows:
            st.markdown("---")
            st.subheader("Summary across all CFU images")
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True)

            # Combined CSV download
            all_combined = pd.concat(all_rows, ignore_index=True)
            combined_buf = io.StringIO()
            all_combined.to_csv(combined_buf, index=False)
            st.download_button(
                label="Download combined CFU_counts.csv",
                data=combined_buf.getvalue(),
                file_name="CFU_counts.csv",
                mime="text/csv",
            )


# ---------- Confluency mode (UI-focused) ----------

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
            "If the mask looks noisy, try a different method or adjust acquisition settings next time."
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

            # ---- UI: per-image results ----
            st.subheader(f"Results — {image_name}")
            st.metric("Percent confluency", f"{percent:.1f}%")

            st.caption("Overlay of detected cell-covered area")
            overlay_img = confluency_overlay_image(image_bgr, mask)
            st.image(
                cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                caption=f"Estimated confluency: {percent:.1f}%",
                use_column_width=True,
            )

            # Optional: per-image CSV download
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                label=f"Download CSV for {image_name}",
                data=csv_buf.getvalue(),
                file_name=f"{Path(image_name).stem}_confluency.csv",
                mime="text/csv",
            )

        # ---- UI: combined table across images ----
        if all_rows:
            st.markdown("---")
            st.subheader("Confluency summary across images")
            combined = pd.concat(all_rows, ignore_index=True)
            st.dataframe(combined, use_container_width=True)

            combined_buf = io.StringIO()
            combined.to_csv(combined_buf, index=False)
            st.download_button(
                label="Download combined confluency_results.csv",
                data=combined_buf.getvalue(),
                file_name="confluency_results.csv",
                mime="text/csv",
            )


# ---------- Main app ----------

def main():
    st.title("Count in a Click — CFU & Confluency Demo")
    st.write(
        "Quick prototype to count colonies and estimate confluency, with on-screen metrics and optional CSV export."
    )

    mode = st.sidebar.radio(
        "Choose scenario",
        ["CFU: Colony Count", "Confluency: % Area Covered"],
    )

    if mode == "CFU: Colony Count":
        run_cfu_mode()
    else:
        run_confluency_mode()


if __name__ == "__main__":
    main()
