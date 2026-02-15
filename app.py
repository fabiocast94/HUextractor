import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import pydicom
from skimage.draw import polygon

st.set_page_config(layout="wide")
st.title("🧠 Production Radiomics ROI Analyzer")

# =====================================================
# CACHE (enorme boost performance)
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
        abs(z_positions[1]-z_positions[0])
    )

    origin = np.array(slices[0].ImagePositionPatient)

    return volume, slices, z_positions, spacing, origin


# =====================================================
# MATCH RTSTRUCT ↔ CT (PRO STYLE)
# =====================================================

def get_referenced_series_uid(rt):

    try:
        return rt.ReferencedFrameOfReferenceSequence[0]\
            .RTReferencedStudySequence[0]\
            .RTReferencedSeriesSequence[0]\
            .SeriesInstanceUID
    except:
        return None


# =====================================================
# ROI helpers
# =====================================================

def get_roi_names(rt):
    return [r.ROIName for r in rt.StructureSetROISequence]


def get_roi_number(rt, roi_name):
    for r in rt.StructureSetROISequence:
        if r.ROIName == roi_name:
            return r.ROINumber
    return None


# =====================================================
# CONTOUR → MASK (research-style)
# =====================================================

def contour_to_mask(rt, roi_name, volume_shape, slices,
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

        pts = np.array(contour.ContourData).reshape(-1,3)

        z = pts[0,2]
        slice_idx = np.argmin(np.abs(z_positions - z))

        rows = (pts[:,1] - origin[1]) / row_spacing
        cols = (pts[:,0] - origin[0]) / col_spacing

        rr, cc = polygon(rows, cols, shape=volume_shape[:2])
        mask[rr, cc, slice_idx] = True

    return mask


# =====================================================
# Upload section
# =====================================================

uploaded_ct = st.file_uploader(
    "Upload CT DICOM (multi patient)",
    accept_multiple_files=True,
    type=["dcm"]
)

uploaded_rt = st.file_uploader(
    "Upload RTSTRUCT",
    accept_multiple_files=True,
    type=["dcm"]
)

if uploaded_ct and uploaded_rt:

    temp_dir = tempfile.mkdtemp()

    # ---------- SAVE FILES ----------
    ct_map = {}
    rt_map = {}

    for f in uploaded_ct:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())

        ds = pydicom.dcmread(path, stop_before_pixels=True)
        uid = ds.SeriesInstanceUID
        ct_map.setdefault(uid, []).append(path)

    for f in uploaded_rt:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())

        rt = pydicom.dcmread(path, stop_before_pixels=True)
        uid = get_referenced_series_uid(rt)
        if uid:
            rt_map[uid] = path

    series_ids = list(set(ct_map) & set(rt_map))

    if len(series_ids) == 0:
        st.error("No matching CT / RTSTRUCT series.")
        st.stop()

    st.success(f"{len(series_ids)} matched datasets found")

    # ROI selection from first dataset
    rt0 = pydicom.dcmread(rt_map[series_ids[0]])
    roi_names = get_roi_names(rt0)

    selected_rois = st.multiselect(
        "Select ROI for analysis",
        roi_names,
        default=roi_names
    )

    # =================================================
    # ANALYSIS
    # =================================================

    if st.button("▶ Run analysis"):

        all_results = []
        progress = st.progress(0)

        for i, uid in enumerate(series_ids):

            volume, slices, z_positions, spacing, origin = \
                load_ct_series(ct_map[uid])

            rt = pydicom.dcmread(rt_map[uid])

            patient_id = slices[0].PatientID

            for roi in selected_rois:

                mask = contour_to_mask(
                    rt,
                    roi,
                    volume.shape,
                    slices,
                    z_positions,
                    spacing,
                    origin
                )

                vals = volume[mask]

                if len(vals) == 0:
                    continue

                all_results.append({
                    "PatientID": patient_id,
                    "SeriesUID": uid,
                    "ROI": roi,
                    "Mean": np.mean(vals),
                    "STD": np.std(vals),
                    "Voxels": len(vals)
                })

            progress.progress((i+1)/len(series_ids))

        df = pd.DataFrame(all_results)

        st.subheader("📊 Results")

        for pid in df.PatientID.unique():
            st.markdown(f"### 👤 Patient {pid}")
            st.dataframe(df[df.PatientID == pid])

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "radiomics_results.csv"
        )
