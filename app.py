import streamlit as st
import zipfile
import os
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import tempfile

st.set_page_config(page_title="Analisi HU CT+RTSTRUCT", layout="wide")
st.title("Analisi HU dalle CT e RTSTRUCT")

# Caricamento file
uploaded_ct = st.file_uploader("Carica cartella CT (zip)", type="zip")
uploaded_rt = st.file_uploader("Carica RTSTRUCT (.dcm)", type="dcm")

if uploaded_ct and uploaded_rt:
    with st.spinner("Estrazione dati e calcolo HU..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            # salva zip CT
            ct_zip_path = os.path.join(tmpdir, "ct.zip")
            with open(ct_zip_path, "wb") as f:
                f.write(uploaded_ct.getbuffer())
            
            # estrai zip
            with zipfile.ZipFile(ct_zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # lista file DICOM CT
            ct_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".dcm")]
            
            if not ct_files:
                st.error("Nessun file DICOM trovato nella cartella CT.")
            else:
                # leggi volume CT
                ct_slices = [pydicom.dcmread(f) for f in ct_files]
                ct_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                ct_volume = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in ct_slices])
                
                # salva RTSTRUCT temporaneo
                rt_path = os.path.join(tmpdir, "rtstruct.dcm")
                with open(rt_path, "wb") as f:
                    f.write(uploaded_rt.getbuffer())
                
                # carica RTSTRUCT
                rtstruct = RTStructBuilder.create_from(dicom_series_path=tmpdir, rt_struct_path=rt_path)
                
                # seleziona ROI disponibile
                roi_names = rtstruct.get_roi_names()
                if not roi_names:
                    st.error("Nessuna ROI trovata nel RTSTRUCT.")
                else:
                    roi_selected = st.selectbox("Seleziona ROI", roi_names)
                    
                    # estrai maschera con resample=True
                    mask = rtstruct.get_roi_mask_by_name(roi_selected, resample=True)
                    
                    # controlla compatibilità shape
                    if mask.shape != ct_volume.shape:
                        st.error(f"Shape non compatibili: CT {ct_volume.shape}, mask {mask.shape}")
                    else:
                        roi_hu = ct_volume[mask.astype(bool)]
                        if roi_hu.size == 0:
                            st.warning("ROI selezionata vuota: nessun voxel trovato.")
                        else:
                            mean_hu = np.mean(roi_hu)
                            std_hu = np.std(roi_hu)
                            st.success(f"Calcolo completato per ROI: **{roi_selected}**")
                            st.write(f"**Valore medio HU:** {mean_hu:.2f}")
                            st.write(f"**Deviazione standard HU:** {std_hu:.2f}")
