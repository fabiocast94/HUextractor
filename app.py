import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd

import pydicom
from rt_utils import RTStructBuilder

st.set_page_config(page_title="RTSTRUCT ROI Analyzer", layout="wide")

st.title("📊 RTSTRUCT ROI Mean & STD Analyzer")

st.markdown("""
Carica:
- Cartella CT (DICOM)
- File RTSTRUCT

L'app calcola media e deviazione standard delle ROI selezionate.
""")

# -------------------------
# Upload multipli pazienti
# -------------------------
uploaded_ct = st.file_uploader(
    "Upload CT DICOM files (anche multipli pazienti)",
    accept_multiple_files=True,
    type=["dcm"]
)

uploaded_rt = st.file_uploader(
    "Upload RTSTRUCT files",
    accept_multiple_files=True,
    type=["dcm"]
)

if uploaded_ct and uploaded_rt:

    # Organizzazione pazienti
    st.info("Parsing DICOM files...")

    # Salvataggio temporaneo
    temp_dir = tempfile.mkdtemp()

    ct_by_patient = {}
    rt_by_patient = {}

    # ---- CT ----
    for file in uploaded_ct:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        ds = pydicom.dcmread(path, stop_before_pixels=True)
        pid = ds.PatientID

        if pid not in ct_by_patient:
            ct_by_patient[pid] = []

        ct_by_patient[pid].append(path)

    # ---- RTSTRUCT ----
    for file in uploaded_rt:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        ds = pydicom.dcmread(path, stop_before_pixels=True)
        pid = ds.PatientID
        rt_by_patient[pid] = path

    common_patients = list(set(ct_by_patient.keys()) & set(rt_by_patient.keys()))

    if not common_patients:
        st.error("Nessun paziente con CT + RTSTRUCT matching.")
        st.stop()

    st.success(f"Trovati {len(common_patients)} pazienti.")

    # =====================
    # ROI Selection
    # =====================
    sample_patient = common_patients[0]

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=os.path.dirname(ct_by_patient[sample_patient][0]),
        rt_struct_path=rt_by_patient[sample_patient]
    )

    roi_names = rtstruct.get_roi_names()

    selected_rois = st.multiselect(
        "Seleziona ROI da analizzare",
        roi_names,
        default=roi_names
    )

    if st.button("▶️ Avvia Analisi"):

        results = []

        progress = st.progress(0)

        for i, pid in enumerate(common_patients):

            try:
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=os.path.dirname(ct_by_patient[pid][0]),
                    rt_struct_path=rt_by_patient[pid]
                )

                for roi in selected_rois:

                    try:
                        mask = rtstruct.get_roi_mask_by_name(roi)

                        # caricamento voxel CT
                        slices = [
                            pydicom.dcmread(f)
                            for f in ct_by_patient[pid]
                        ]
                        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

                        volume = np.stack([s.pixel_array for s in slices], axis=-1)

                        values = volume[mask]

                        mean_val = np.mean(values)
                        std_val = np.std(values)

                        results.append({
                            "PatientID": pid,
                            "ROI": roi,
                            "Mean": mean_val,
                            "STD": std_val,
                            "N_voxels": len(values)
                        })

                    except Exception as e:
                        st.warning(f"Errore ROI {roi} paziente {pid}: {e}")

            except Exception as e:
                st.error(f"Errore paziente {pid}: {e}")

            progress.progress((i + 1) / len(common_patients))

        df = pd.DataFrame(results)

        st.subheader("📈 Risultati")

        for pid in df["PatientID"].unique():
            st.markdown(f"### 🧑 Paziente {pid}")
            st.dataframe(df[df["PatientID"] == pid])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV risultati",
            csv,
            "roi_analysis.csv",
            "text/csv"
        )
