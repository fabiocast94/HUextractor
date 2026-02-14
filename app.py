import streamlit as st
import zipfile
import os
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import tempfile
from scipy.ndimage import zoom

st.set_page_config(page_title="Analisi HU CT+RTSTRUCT", layout="wide")
st.title("Analisi HU dalle CT e RTSTRUCT")

# 🔹 Caricamento file
uploaded_ct = st.file_uploader("Carica cartella CT (zip)", type="zip")
uploaded_rt = st.file_uploader("Carica RTSTRUCT (.dcm)", type="dcm")

# Session state per salvare ROI selezionate e file caricati
if "roi_selected" not in st.session_state:
    st.session_state.roi_selected = []
if "ct_files" not in st.session_state:
    st.session_state.ct_files = None
if "rt_path" not in st.session_state:
    st.session_state.rt_path = None

roi_names = []

if uploaded_ct and uploaded_rt:
    # 🔹 Estrazione solo dei nomi delle ROI senza fare calcoli pesanti
    with tempfile.TemporaryDirectory() as tmpdir:
        # salva zip CT temporaneo
        ct_zip_path = os.path.join(tmpdir, "ct.zip")
        with open(ct_zip_path, "wb") as f:
            f.write(uploaded_ct.getbuffer())
        with zipfile.ZipFile(ct_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # lista file DICOM CT
        st.session_state.ct_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".dcm")]
        
        # salva RTSTRUCT
        st.session_state.rt_path = os.path.join(tmpdir, "rtstruct.dcm")
        with open(st.session_state.rt_path, "wb") as f:
            f.write(uploaded_rt.getbuffer())
        
        # carica RTSTRUCT solo per leggere ROI names
        rtstruct = RTStructBuilder.create_from(dicom_series_path=tmpdir, rt_struct_path=st.session_state.rt_path)
        roi_names = rtstruct.get_roi_names()

# 🔹 Selezione ROI
if roi_names:
    st.session_state.roi_selected = st.multiselect(
        "Seleziona ROI da analizzare",
        roi_names,
        default=[]
    )

# 🔹 Bottone per eseguire calcolo
if st.button("Esegui analisi") and st.session_state.roi_selected:
    if not st.session_state.ct_files or not st.session_state.rt_path:
        st.error("Errore: file CT o RTSTRUCT non disponibili.")
    else:
        with st.spinner("Estrazione dati e calcolo HU..."):
            # leggi volume CT
            ct_slices = [pydicom.dcmread(f) for f in st.session_state.ct_files]
            ct_slices.sort(key=lambda x: float(getattr(x, "ImagePositionPatient", [0,0,0])[2]))
            ct_volume = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in ct_slices])
            
            # carica RTSTRUCT
            tmpdir = os.path.dirname(st.session_state.rt_path)
            rtstruct = RTStructBuilder.create_from(dicom_series_path=tmpdir, rt_struct_path=st.session_state.rt_path)
            
            results = []
            sample_mask = rtstruct.get_roi_mask_by_name(st.session_state.roi_selected[0])
            factors = (
                ct_volume.shape[0] / sample_mask.shape[0],
                ct_volume.shape[1] / sample_mask.shape[1],
                ct_volume.shape[2] / sample_mask.shape[2]
            )
            
            for roi in st.session_state.roi_selected:
                mask = rtstruct.get_roi_mask_by_name(roi)
                mask_resampled = zoom(mask.astype(float), factors, order=0).astype(bool)
                
                if mask_resampled.shape != ct_volume.shape:
                    st.warning(f"Shape non compatibili per ROI {roi}, saltata.")
                    continue
                
                roi_hu = ct_volume[mask_resampled]
                if roi_hu.size == 0:
                    st.warning(f"ROI {roi} vuota, saltata.")
                    continue
                
                results.append({
                    "ROI": roi,
                    "Mean_HU": np.mean(roi_hu),
                    "Std_HU": np.std(roi_hu)
                })
            
            if results:
                st.dataframe(results)
                st.success("Analisi completata!")
            else:
                st.warning("Nessun dato valido da analizzare.")
