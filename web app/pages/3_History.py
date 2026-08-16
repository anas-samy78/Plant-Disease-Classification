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

st.markdown('<div class="title-container"><h1>Your identification history</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Everything you\'ve identified this session</p>', unsafe_allow_html=True)
st.markdown("---")
 
if "history" not in st.session_state:
    st.session_state["history"] = []
 
if len(st.session_state["history"]) == 0:
    st.info("You haven't identified any plants yet. Head to the Predict page to get started.")
else:

    for item in reversed(st.session_state["history"]):
        col1, col2 = st.columns([1, 3])
 
        with col1:
            st.image(item["image"], use_container_width=True)
 
        with col2:
            st.markdown(f"**{item['species']}**")
            st.write(f"Confidence: {item['confidence']:.1f}%")
 
        st.markdown("---")
 
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()
 