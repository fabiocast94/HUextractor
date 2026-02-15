import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import pydicom
from skimage.draw import polygon

st.set_page_config(page_title="ROI HU Analyzer", layout="wide")
st.title("🧠 Radiomics ROI Analyzer (Stable Version)")

# =====================================================
# Utility functions
# =====================================================

def load_ct_series(files):
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices], axis=-1)

    spacing = (
        float(slices[0].PixelSpacing[0]),
        float(slices[0].PixelSpacing[1]),
        float(slices[0].SliceThickness),
    )

    origin = np.array(slices[0].ImagePositionPatient)

    return volume, slices, spacing, origin


def patient_roi_names(rt):
    names = []
    for roi in rt.StructureSetROISequence:
        names.append(roi.ROIName)
    return names


def build_mask_from_contours(rt, roi_name, volume_shape, slices):

    mask = np.zeros(volume_shape, dtype=bool)

    # map roi number
    roi_number = None
    for roi in rt.StructureSetROISequence:
        if roi.ROIName == roi_name:
            roi_number = roi.ROINumber

    if roi_number is None:
        return mask

    contours = None
    for c in rt.ROIContourSequence:
        if c.ReferencedROINumber == roi_number:
            contours = c

    if contours is None:
        return mask

    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]

    for contour in contours.ContourSequence:

        pts = np.array(contour.ContourData).reshape(-1,3)

        z = pts[0,2]
        slice_idx = np.argmin(np.abs(np.array(z_positions)-z))

        row_spacing = float(slices[0].PixelSpacing[0])
        col_spacing = float(slices[0].PixelSpacing[1])

        origin = np.array(slices[0].ImagePositionPatient)

        rows = (pts[:,1] - origin[1]) / row_spacing
        cols = (pts[:,0] - origin[0]) / col_spacing

        rr, cc = polygon(rows, cols, shape=volume_shape[:2])
        mask[rr, cc, slice_idx] = True

    return mask


# =====================================================
# Upload
# =====================================================

uploaded_ct = st.file_uploader(
    "Upload CT DICOM (multi-patient)",
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

    ct_by_patient = {}
    rt_by_patient = {}

    # ---- CT ----
    for f in uploaded_ct:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())

        ds = pydicom.dcmread(path, stop_before_pixels=True)
        pid = ds.PatientID

        ct_by_patient.setdefault(pid, []).append(path)

    # ---- RT ----
    for f in uploaded_rt:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())

        ds = pydicom.dcmread(path, stop_before_pixels=True)
        rt_by_patient[ds.PatientID] = path

    patients = list(set(ct_by_patient) & set(rt_by_patient))

    if not patients:
        st.error("No matching patients.")
        st.stop()

    st.success(f"{len(patients)} patients detected")

    # ROI selection (first patient)
    rt_sample = pydicom.dcmread(rt_by_patient[patients[0]])
    roi_names = patient_roi_names(rt_sample)

    selected_rois = st.multiselect(
        "Select ROI",
        roi_names,
        default=roi_names
    )

    if st.button("▶ Start analysis"):

        results = []
        progress = st.progress(0)

        for i, pid in enumerate(patients):

            volume, slices, _, _ = load_ct_series(ct_by_patient[pid])
            rt = pydicom.dcmread(rt_by_patient[pid])

            for roi in selected_rois:

                mask = build_mask_from_contours(
                    rt,
                    roi,
                    volume.shape,
                    slices
                )

                vals = volume[mask]

                if len(vals) == 0:
                    continue

                results.append({
                    "PatientID": pid,
                    "ROI": roi,
                    "Mean": np.mean(vals),
                    "STD": np.std(vals),
                    "N_voxels": len(vals)
                })

            progress.progress((i+1)/len(patients))

        df = pd.DataFrame(results)

        st.subheader("Results")

        for pid in df.PatientID.unique():
            st.markdown(f"### Patient {pid}")
            st.dataframe(df[df.PatientID==pid])

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "roi_results.csv"
        )
