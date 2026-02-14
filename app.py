import streamlit as st
import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import pydicom
from rt_utils import RTStructBuilder
import re

# ======================================
# STREAMLIT UI
# ======================================

st.title("HU ROI Extractor (Eclipse RTSTRUCT)")
st.write("Upload uno ZIP con i pazienti DICOM.")

uploaded_file = st.file_uploader(
    "Carica DATA.zip",
    type=["zip"]
)

# ======================================
# FUNCTIONS
# ======================================

ROI_RENAME_RULES = {
    "PTV": "PTV",
    "CTV": "CTV",
    "GTV": "GTV",
    "LUNG_L": "LUNG_L",
    "LUNG_R": "LUNG_R"
}


def normalize_roi_name(name):
    """Normalizza i nomi delle ROI usando regex."""
    name_up = name.upper()
    for key in ROI_RENAME_RULES:
        if re.search(rf"\b{key}\b", name_up):
            return ROI_RENAME_RULES[key]
    return name


def load_ct_hu(ct_folder):
    """Carica CT e restituisce volume in HU."""
    files = [
        pydicom.dcmread(os.path.join(ct_folder, f))
        for f in os.listdir(ct_folder)
        if not f.startswith(".")
    ]

    files.sort(key=lambda x: float(getattr(x, "ImagePositionPatient", [0, 0, 0])[2]))

    slope = float(getattr(files[0], "RescaleSlope", 1))
    intercept = float(getattr(files[0], "RescaleIntercept", 0))

    volume = np.stack([f.pixel_array for f in files], axis=2).astype(np.float32)
    return volume * slope + intercept


def compute_stats(values):
    """Calcola statistiche HU per un array di valori."""
    return {
        "MeanHU": float(np.mean(values)),
        "StdHU": float(np.std(values)),
        "MedianHU": float(np.median(values)),
        "P05HU": float(np.percentile(values, 5)),
        "P95HU": float(np.percentile(values, 95)),
        "VoxelCount": int(len(values))
    }


def process_patient(patient_path):
    """Elabora un singolo paziente: CT + RTSTRUCT."""
    patient_id = os.path.basename(patient_path)
    ct_folder = os.path.join(patient_path, "CT")
    rtstruct_path = os.path.join(patient_path, "RTSTRUCT.dcm")

    if not (os.path.exists(ct_folder) and os.path.exists(rtstruct_path)):
        st.warning(f"Paziente {patient_id}: CT o RTSTRUCT mancanti")
        return []

    hu = load_ct_hu(ct_folder)

    try:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=ct_folder,
            rt_struct_path=rtstruct_path
        )
    except Exception as e:
        st.warning(f"Paziente {patient_id}: impossibile caricare RTSTRUCT ({e})")
        return []

    results = []

    for roi in rtstruct.get_roi_names():
        try:
            mask = rtstruct.get_roi_mask_by_name(roi)
            if mask.shape != hu.shape:
                st.warning(f"Paziente {patient_id}, ROI {roi}: shape mask diversa da CT")
                continue

            roi_hu = hu[mask]
            if roi_hu.size == 0:
                continue

            row = {
                "Patient": patient_id,
                "ROI_Original": roi,
                "ROI_Normalized": normalize_roi_name(roi),
            }
            row.update(compute_stats(roi_hu))
            results.append(row)
        except Exception as e:
            st.warning(f"Paziente {patient_id}, ROI {roi}: errore {e}")
            continue

    return results

# ======================================
# MAIN LOGIC
# ======================================

if uploaded_file:

    st.info("Estrazione file ZIP...")

    with tempfile.TemporaryDirectory() as tmpdir:

        zip_path = os.path.join(tmpdir, "data.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Trova cartella DATA
        data_root = next(
            (os.path.join(root, d)
             for root, dirs, _ in os.walk(tmpdir)
             for d in dirs if d.upper() == "DATA"),
            None
        )

        if data_root is None:
            st.error("Cartella DATA non trovata nello ZIP.")
            st.stop()

        patients = [
            os.path.join(data_root, p)
            for p in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, p))
        ]

        st.write(f"Trovati {len(patients)} pazienti")

        all_results = []
        progress = st.progress(0)

        for i, p in enumerate(patients):
            all_results.extend(process_patient(p))
            progress.progress(int((i + 1) / len(patients) * 100))

        df = pd.DataFrame(all_results)

        if len(df) == 0:
            st.error("Nessun risultato trovato.")
        else:
            st.success("Calcolo completato!")

            df = df.sort_values(by=["Patient", "ROI_Normalized"])
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                "HU_results.csv",
                "text/csv"
            )
