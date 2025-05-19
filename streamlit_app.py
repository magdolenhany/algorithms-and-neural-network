import streamlit as st
from modules import pca_module, genetic_module, art_module,rbf_module, som_module

st.set_page_config(page_title="PCA + Genetic + ART App", layout="centered")
st.title("🧠 PCA, Genetic Algorithm, and ART Simulation")

tab1, tab2, tab3 ,tab4, tab5= st.tabs(["📊 PCA", "🧬 Genetic Algorithm", "🧠 ART Network","🔴 RBF", "🔵 SOM"])

with tab1:
    pca_module.run_pca()

with tab2:
    genetic_module.run_genetic()

with tab3:
    art_module.run_art()
with tab4:
    rbf_module.run_rbf()

with tab5:
    som_module.run_som()