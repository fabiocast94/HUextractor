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
st.write("Upload uno ZIP con cartelle pazienti DICOM.")

uploaded_file = st.file_uploader(
    "Carica ZIP con cartelle pazienti",
    type=["zip"]
)

# ======================================
# ROI NORMALIZATION
# ======================================

ROI_RENAME_RULES = {
    "PTV": ["PTV", "PTV1", "PTV_BOOST"],
    "CTV": ["CTV", "CTV1"],
    "GTV": ["GTV", "GTV_PRIMARY"],
    "LUNG_L": ["LUNG_L", "LEFT_LUNG"],
    "LUNG_R": ["LUNG_R", "RIGHT_LUNG"],
    "HEART": ["HEART", "COR"],
    "SPINAL_CORD": ["SPINALCORD", "SC", "SPINE"],
    "ESOPHAGUS": ["ESOPHAGUS", "ESO"]
}

def normalize_roi_name(name):
    name_up = name.upper()
    for normalized, aliases in ROI_RENAME_RULES.items():
        for alias in aliases:
            if re.search(rf"\b{alias.upper()}\b", name_up):
                return normalized
    return name

# ======================================
# FUNCTIONS
# ======================================

def load_ct_hu(ct_folder):
    """Carica CT e restituisce volume in HU, gestendo slice mancanti o tag assenti."""
    files = [
        pydicom.dcmread(os.path.join(ct_folder, f))
        for f in os.listdir(ct_folder)
        if not f.startswith(".") and f.lower().endswith(".dcm")
    ]
    
    # Ordinamento: prima ImagePositionPatient, se manca InstanceNumber
    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
        else:
            return 0.0

    files.sort(key=sort_key)

    # Controllo slice dimensioni
    shapes = [f.pixel_array.shape for f in files]
    if len(set(shapes)) > 1:
        st.warning(f"Slice CT con dimensioni diverse trovate: {set(shapes)}. Verranno ignorate slice anomale.")
        # tenere solo le slice con shape più comune
        from collections import Counter
        most_common_shape = Counter(shapes).most_common(1)[0][0]
        files = [f for f in files if f.pixel_array.shape == most_common_shape]

    slope = float(getattr(files[0], "RescaleSlope", 1))
    intercept = float(getattr(files[0], "RescaleIntercept", 0))

    volume = np.stack([f.pixel_array for f in files], axis=2).astype(np.float32)
    return volume * slope + intercept

def compute_stats(values):
    return {
        "MeanHU": float(np.mean(values)),
        "StdHU": float(np.std(values)),
        "MedianHU": float(np.median(values)),
        "P05HU": float(np.percentile(values, 5)),
        "P95HU": float(np.percentile(values, 95)),
        "VoxelCount": int(len(values))
    }

def find_ct_folder(patient_path):
    for d in os.listdir(patient_path):
        d_path = os.path.join(patient_path, d)
        if os.path.isdir(d_path) and any(f.lower().endswith(".dcm") for f in os.listdir(d_path)):
            return d_path
    return None

def find_rtstruct(patient_path):
    for f in os.listdir(patient_path):
        if "RTSTRUCT" in f.upper() and f.lower().endswith(".dcm"):
            return os.path.join(patient_path, f)
    return None

def process_patient(patient_path):
    patient_id = os.path.basename(patient_path)
    ct_folder = find_ct_folder(patient_path)
    rtstruct_path = find_rtstruct(patient_path)

    if ct_folder is None or rtstruct_path is None:
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
                st.warning(f"Paziente {patient_id}, ROI {roi}: shape mask {mask.shape} diversa da CT {hu.shape}")
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

        patients = [
            os.path.join(root, d)
            for root, dirs, _ in os.walk(tmpdir)
            for d in dirs
        ]

        if len(patients) == 0:
            st.error("Nessuna cartella paziente trovata nello ZIP.")
            st.stop()

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
            st.download_button("Download CSV", csv, "HU_results.csv", "text/csv")
