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

# --- Caricamento file ---
uploaded_ct = st.file_uploader("Carica cartella CT (zip)", type="zip")
uploaded_rt = st.file_uploader("Carica RTSTRUCT (.dcm)", type="dcm")

if uploaded_ct and uploaded_rt:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Salva zip CT e RTSTRUCT
        ct_zip_path = os.path.join(tmpdir, "ct.zip")
        with open(ct_zip_path, "wb") as f:
            f.write(uploaded_ct.getbuffer())
        with zipfile.ZipFile(ct_zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        rt_path = os.path.join(tmpdir, "rtstruct.dcm")
        with open(rt_path, "wb") as f:
            f.write(uploaded_rt.getbuffer())
        
        # Lista file DICOM CT
        ct_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".dcm")]
        if not ct_files:
            st.error("Nessun file DICOM trovato nella cartella CT.")
        else:
            # Carica volume CT
            ct_slices = [pydicom.dcmread(f) for f in ct_files]
            ct_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            ct_volume = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in ct_slices])
            
            # Carica RTSTRUCT
            rtstruct = RTStructBuilder.create_from(dicom_series_path=tmpdir, rt_struct_path=rt_path)
            roi_names = rtstruct.get_roi_names()
            
            if not roi_names:
                st.error("Nessuna ROI trovata nel RTSTRUCT.")
            else:
                # Multiselect ROI
                roi_selected = st.multiselect(
                    "Seleziona ROI da analizzare",
                    roi_names,
                    default=roi_names
                )
                
                # Bottone per eseguire analisi
                if st.button("Esegui Analisi"):
                    with st.spinner("Calcolo HU per le ROI selezionate..."):
                        results = []
                        # fattori di resample usando prima ROI selezionata
                        sample_mask = rtstruct.get_roi_mask_by_name(roi_selected[0])
                        factors = (
                            ct_volume.shape[0] / sample_mask.shape[0],
                            ct_volume.shape[1] / sample_mask.shape[1],
                            ct_volume.shape[2] / sample_mask.shape[2]
                        )
                        
                        for roi in roi_selected:
                            mask = rtstruct.get_roi_mask_by_name(roi)
                            mask_resampled = zoom(mask.astype(float), factors, order=0).astype(bool)
                            
                            if mask_resampled.shape != ct_volume.shape:
                                st.warning(f"Shape non compatibili per ROI {roi}, saltata.")
                                continue
                            
                            roi_hu = ct_volume[mask_resampled]
                            if roi_hu.size == 0:
                                st.warning(f"ROI {roi} vuota: nessun voxel trovato, saltata.")
                                continue
                            
                            results.append({
                                "ROI": roi,
                                "Mean_HU": np.mean(roi_hu),
                                "Std_HU": np.std(roi_hu),
                                "Min_HU": np.min(roi_hu),
                                "Max_HU": np.max(roi_hu)
                            })
                        
                        if results:
                            # Visualizza risultati a video
                            st.dataframe(results)
                            st.success("Analisi completata!")
                        else:
                            st.warning("Nessun dato valido da analizzare.")
