import streamlit as st
import pandas as pd
from PIL import Image
import timm
import torch
import os

st.markdown("""
<style>

.main {
    background-color: #F7F5EF;
}

.stApp {
    background: linear-gradient(180deg, #F7F5EF 0%, #EEEBE0 100%);
}

/* ===========================
   HEADER
=========================== */

header[data-testid="stHeader"] {
    background-color: #FBFAF6;
    border-bottom: 1px solid #DDD8C8;
}

[data-testid="stToolbar"] {
    background-color: transparent;
}

[data-testid="stDecoration"] {
    background: linear-gradient(90deg, #4A7C59, #2E5339);
}

/* ===========================
   TITLE
=========================== */

.title-container {
    text-align: center;
    padding: 1.2rem 0 .3rem;
}

.title-container h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg,#4A7C59,#2E5339);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:0;
}

.subtitle {
    text-align:center;
    color:#6B7266;
    font-size:1.05rem;
    margin-top:-.3rem;
    margin-bottom:1.5rem;
}

/* ===========================
   CARDS
=========================== */

.section-card {

    background:#FFFFFF;

    border:1px solid #E2DDD0;

    border-radius:16px;

    padding:1.5rem;

    margin-bottom:1rem;

}

.section-title{

    color:#2E5339;

    font-weight:700;

    font-size:1.2rem;

}

/* ===========================
   TEXT
=========================== */

p,
label,
.stMarkdown{

    color:#3A3D35;

}

/* ===========================
   SIDEBAR
=========================== */

[data-testid="stSidebar"]{

    background:#F0EDE2;

    border-right:1px solid #DDD8C8;

}

[data-testid="stSidebar"] *{

    color:#3A3D35 !important;

}

/* ===========================
   CAPTION
=========================== */

[data-testid="stCaptionContainer"]{

    color:#8A8D82 !important;

}

/* ===========================
   INPUTS
=========================== */

.stSelectbox label,
.stNumberInput label{

    color:#3A3D35 !important;

    font-weight:600;

}

div[data-baseweb="select"]>div{

    background:#FFFFFF;

    border:1px solid #DDD8C8;

    color:#3A3D35;

}

div[data-testid="stNumberInput"] input{

    background:#FFFFFF;

    color:#3A3D35;

    border:1px solid #DDD8C8;

    border-radius:8px;

}

div[data-testid="stNumberInput"] input:focus{

    border:1px solid #4A7C59;

    box-shadow:0 0 0 1px #4A7C5955;

}

div[data-testid="stNumberInput"] button{

    background:#F0EDE2;

    border:1px solid #DDD8C8;

    color:#3A3D35;

}

div[data-testid="stNumberInput"] button:hover{

    background:#E2DDD0;

    color:#2E5339;

}

/* ===========================
   BUTTON
=========================== */

div.stButton>button{

    width:100%;

    background:linear-gradient(90deg,#4A7C59,#2E5339);

    color:white;

    font-size:1.05rem;

    font-weight:700;

    padding:.75rem;

    border:none;

    border-radius:12px;

    transition:.25s;

}

div.stButton>button:hover{

    transform:scale(1.02);

    box-shadow:0 6px 20px rgba(46,83,57,.3);

}

/* ===========================
   RESULT CARD
=========================== */

.result-card{

    background:#FFFFFF;

    border:1px solid #E2DDD0;

    border-radius:18px;

    padding:2rem;

    text-align:center;

    margin-top:1rem;

}

.result-card h2{

    color:#8A8D82;

    font-size:1rem;

    margin-bottom:.5rem;

}

.result-card h1{

    font-size:3rem;

    font-weight:800;

    color:#2E5339;

    margin:0;

}

/* ===========================
   METRICS
=========================== */

[data-testid="metric-container"]{

    background:#FFFFFF;

    border:1px solid #E2DDD0;

    border-radius:15px;

    padding:18px;

}

/* ===========================
   PROGRESS BAR
=========================== */

.stProgress > div > div > div{

    background:linear-gradient(90deg,#8BAF6F,#4A7C59);

}

/* ===========================
   DATAFRAME
=========================== */

[data-testid="stDataFrame"]{

    border-radius:15px;

    overflow:hidden;

}

/* ===========================
   ALERTS
=========================== */

/* سيب Streamlit يتحكم في ألوانهم */
div[data-baseweb="notification"]{

    border-radius:12px;

}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-container"><h1>About PlantID</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">How this project works</p>', unsafe_allow_html=True)
st.markdown("---")

 
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">The problem</div>', unsafe_allow_html=True)
st.write(
    "Identifying plants accurately can be difficult, especially for beginners "
    "or when a plant shows early signs of disease. PlantID makes this fast "
    "and accessible for gardeners, farmers, and plant enthusiasts alike."
)
st.markdown('</div>', unsafe_allow_html=True)
 
 
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
st.write(
    "PlantID uses two specialized models. First, DINOv2, a vision transformer "
    "fine-tuned on the PlantCLEF dataset covering 7,806 plant species, identifies "
    "the plant. Then, a dedicated disease classification model checks the same "
    "photo for common diseases in supported crops."
)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Species", value="7,806")
with col2:
    st.metric(label="Training images", value="1.4M+")
with col3:
    st.metric(label="Results", value="Instant")
 
st.write("")  

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Disease detection scope</div>', unsafe_allow_html=True)
st.write(
    "Disease detection is currently trained on common crops such as apple, "
    "tomato, and corn. Results for other species may be less accurate and "
    "should not replace expert diagnosis."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Built with</div>', unsafe_allow_html=True)
st.write("PyTorch · timm · Transformers · Streamlit")
st.markdown('</div>', unsafe_allow_html=True)
