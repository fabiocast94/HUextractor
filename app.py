import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from skimage.draw import polygon

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="Radiomics ROI Analyzer", layout="wide")
st.title("🧠 Radiomics ROI Analyzer (ZIP Edition)")

st.markdown("""
Upload uno **ZIP** contenente CT DICOM + RTSTRUCT.
L'app calcola Mean e STD HU per ROI selezionate.
""")

# =====================================================
# CACHE CT LOADING (performance boost)
# =====================================================

@st.cache_data(show_spinner=False)
def load_ct_series(files):

    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices], axis=-1)

    z_positions = np.array([
        float(s.ImagePositionPatient[2]) for s in slices
    ])

    spacing = (
        float(slices[0].PixelSpacing[0]),
        float(slices[0].PixelSpacing[1]),
        abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else 1.0
    )

    origin = np.array(slices[0].ImagePositionPatient)

    return volume, slices, z_positions, spacing, origin


# =====================================================
# RTSTRUCT HELPERS
# =====================================================

def get_referenced_series_uid(rt):
    try:
        return rt.ReferencedFrameOfReferenceSequence[0] \
            .RTReferencedStudySequence[0] \
            .RTReferencedSeriesSequence[0] \
            .SeriesInstanceUID
    except:
        return None


def get_roi_names(rt):
    return [r.ROIName for r in rt.StructureSetROISequence]


def get_roi_number(rt, roi_name):
    for r in rt.StructureSetROISequence:
        if r.ROIName == roi_name:
            return r.ROINumber
    return None


# =====================================================
# CONTOUR -> MASK
# =====================================================

def contour_to_mask(rt, roi_name, volume_shape,
                    z_positions, spacing, origin):

    mask = np.zeros(volume_shape, dtype=bool)

    roi_number = get_roi_number(rt, roi_name)
    if roi_number is None:
        return mask

    roi_contours = None
    for rc in rt.ROIContourSequence:
        if rc.ReferencedROINumber == roi_number:
            roi_contours = rc

    if roi_contours is None:
        return mask

    row_spacing, col_spacing, _ = spacing

    for contour in roi_contours.ContourSequence:

        pts = np.array(contour.ContourData).reshape(-1, 3)

        z = pts[0, 2]
        slice_idx = np.argmin(np.abs(z_positions - z))

        rows = (pts[:, 1] - origin[1]) / row_spacing
        cols = (pts[:, 0] - origin[0]) / col_spacing

        rr, cc = polygon(rows, cols, shape=volume_shape[:2])
        mask[rr, cc, slice_idx] = True

    return mask


# =====================================================
# ZIP UPLOAD
# =====================================================

uploaded_zip = st.file_uploader(
    "📦 Upload ZIP (CT + RTSTRUCT)",
    type=["zip"]
)

if uploaded_zip:

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "dataset.zip")

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    st.success("ZIP extracted successfully ✔")

    # =================================================
    # SCAN DICOM FILES
    # =================================================

    ct_map = {}
    rt_map = {}

    all_files = list(Path(temp_dir).rglob("*"))
    dicom_files = [f for f in all_files if f.is_file()]

    progress = st.progress(0)

    for i, file in enumerate(dicom_files):

        try:
            ds = pydicom.dcmread(str(file), stop_before_pixels=True)

            if ds.Modality == "CT":
                uid = ds.SeriesInstanceUID
                ct_map.setdefault(uid, []).append(str(file))

            elif ds.Modality == "RTSTRUCT":
                uid = get_referenced_series_uid(ds)
                if uid:
                    rt_map[uid] = str(file)

        except:
            pass

        progress.progress((i + 1) / len(dicom_files))

    series_ids = list(set(ct_map) & set(rt_map))

    if len(series_ids) == 0:
        st.error("❌ No matching CT / RTSTRUCT found.")
        st.stop()

    st.success(f"✅ Found {len(series_ids)} matched datasets")

    # =================================================
    # ROI SELECTION
    # =================================================

    rt0 = pydicom.dcmread(rt_map[series_ids[0]])
    roi_names = get_roi_names(rt0)

    selected_rois = st.multiselect(
        "Select ROI to analyze",
        roi_names,
        default=roi_names
    )

    # =================================================
    # ANALYSIS
    # =================================================

    if st.button("▶ Run Analysis"):

        results = []
        progress = st.progress(0)

        for i, uid in enumerate(series_ids):

            volume, slices, z_positions, spacing, origin = \
                load_ct_series(ct_map[uid])

            rt = pydicom.dcmread(rt_map[uid])

            patient_id = getattr(slices[0], "PatientID", "Unknown")

            for roi in selected_rois:

                mask = contour_to_mask(
                    rt,
                    roi,
                    volume.shape,
                    z_positions,
                    spacing,
                    origin
                )

                vals = volume[mask]

                if len(vals) == 0:
                    continue

                results.append({
                    "PatientID": patient_id,
                    "SeriesUID": uid,
                    "ROI": roi,
                    "Mean": float(np.mean(vals)),
                    "STD": float(np.std(vals)),
                    "N_voxels": int(len(vals))
                })

            progress.progress((i + 1) / len(series_ids))

        df = pd.DataFrame(results)

        st.subheader("📊 Results")

        for pid in df.PatientID.unique():
            st.markdown(f"### 👤 Patient {pid}")
            st.dataframe(df[df.PatientID == pid],
                         use_container_width=True)

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "radiomics_results.csv",
            "text/csv"
        )
