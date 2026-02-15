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
st.title("🧠 Radiomics ROI Analyzer")

st.markdown("""
Puoi caricare:

- 📦 **un solo ZIP** con tutti i pazienti
- 📦📦 **più ZIP**, uno per paziente

L'app calcola Mean, STD, Min e Max HU per le ROI selezionate.
""")

# =====================================================
# CACHE CT LOADING
# =====================================================
@st.cache_data(show_spinner=False)
def load_ct_series(files):
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    z_positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    spacing = (
        float(slices[0].PixelSpacing[0]),
        float(slices[0].PixelSpacing[1]),
        abs(z_positions[1]-z_positions[0]) if len(z_positions) > 1 else 1.0
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
# MULTI ZIP UPLOAD
# =====================================================
uploaded_zips = st.file_uploader(
    "📦 Upload uno o più ZIP",
    type=["zip"],
    accept_multiple_files=True
)

if uploaded_zips:

    temp_dir = tempfile.mkdtemp()
    st.info(f"Estrazione di {len(uploaded_zips)} ZIP...")

    # Extract all ZIPs
    for i, zip_file in enumerate(uploaded_zips):
        zip_path = os.path.join(temp_dir, f"dataset_{i}.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        extract_dir = os.path.join(temp_dir, f"zip_{i}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    st.success("ZIP estratti ✔")

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
        st.error("❌ Nessun match CT / RTSTRUCT trovato.")
        st.stop()
    st.success(f"✅ Dataset trovati: {len(series_ids)}")

    # =================================================
    # ROI SELECTION MULTI-PATIENT
    # =================================================
    all_roi_names = set()
    for uid in series_ids:
        rt = pydicom.dcmread(rt_map[uid])
        all_roi_names.update(get_roi_names(rt))
    all_roi_names = sorted(list(all_roi_names))

    selected_rois = st.multiselect(
        "Seleziona ROI (unione di tutti i pazienti)",
        all_roi_names,
        default=all_roi_names
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
                    "Min": float(np.min(vals)),
                    "Max": float(np.max(vals)),
                    "N_voxels": int(len(vals))
                })

            progress.progress((i + 1) / len(series_ids))

        df = pd.DataFrame(results)

        st.subheader("📊 Results")
        for pid in df.PatientID.unique():
            sub = df[df.PatientID == pid]
            st.markdown(f"### 👤 Patient {pid}")
            st.dataframe(sub, use_container_width=True)

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "radiomics_results.csv",
            "text/csv"
        )
