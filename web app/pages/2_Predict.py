import streamlit as st
import pandas as pd
from PIL import Image
import timm
import torch
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchvision.transforms as T
from gemini_plant_analysis import get_plant_analysis
from huggingface_hub import hf_hub_download

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

st.markdown('<div class="title-container"><h1>Identify your plant and its condition</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a clear photo of a leaf or plant for the most accurate result</p>', unsafe_allow_html=True)
st.markdown("---")
 
@st.cache_resource
def load_disease_model():
    disease_model = AutoModelForImageClassification.from_pretrained(
        "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    )
    disease_model.eval()
 
    disease_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    return disease_transform, disease_model
 
 
disease_transform, disease_model = load_disease_model()
 
@st.cache_resource
def load_model_and_mappings():
    CLASS_MAPPING_PATH = "class_mapping.txt"
    SPECIES_MAPPING_PATH = "species_id_to_name.txt"
    PRETRAINED_PATH = hf_hub_download(
        repo_id="vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights",
        filename="vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors")
 
    def load_class_mapping(class_list_file):
        with open(class_list_file) as f:
            return {i: line.strip() for i, line in enumerate(f)}
 
    def load_species_mapping(species_map_file):
        df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
        df = df.set_index('species_id')
        return df['species'].to_dict()
 
    cid_to_spid = load_class_mapping(CLASS_MAPPING_PATH)
    spid_to_sp = load_species_mapping(SPECIES_MAPPING_PATH)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=len(cid_to_spid),
        checkpoint_path=PRETRAINED_PATH
    )
    model = model.to(device)
    model = model.eval()
 
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
 
    return model, transforms, cid_to_spid, spid_to_sp, device
 
 
model, transforms, cid_to_spid, spid_to_sp, device = load_model_and_mappings()
 
uploaded_file = st.file_uploader(
    "Drag and drop a photo here, or click to browse",
    type=["jpg", "jpeg", "png"]
)
 
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.5])
 
    with col1:
        st.image(uploaded_file, caption="Your uploaded plant", use_container_width=True)
 
    img = Image.open(uploaded_file)

    with st.spinner("Checking for diseases..."):
        disease_input = disease_transform(img.convert("RGB")).unsqueeze(0)
 
        with torch.no_grad():
            disease_outputs = disease_model(pixel_values=disease_input)
            predicted_disease_id = disease_outputs.logits.argmax(dim=-1).item()
 
        predicted_disease = disease_model.config.id2label[predicted_disease_id]
 
    st.markdown("### Disease check")
    st.warning(f"Detected: {predicted_disease}")
    st.caption("Disease detection currently supports Apple, Tomato, Corn, and other common crops. Not a substitute for expert diagnosis.")
 
    with st.spinner("Analyzing your photo..."):
        img_transformed = transforms(img).unsqueeze(0).to(device)
 
        with torch.no_grad():
            output = model(img_transformed)
            top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
 
        top5_probabilities = top5_probabilities.cpu().detach().numpy()[0]
        top5_class_indices = top5_class_indices.cpu().detach().numpy()[0]
 
    results = []
    for proba, cid in zip(top5_probabilities, top5_class_indices):
        species_id = cid_to_spid[cid]
        species = spid_to_sp[species_id]
        results.append((species, proba))
 
    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<h2>Top match</h2>', unsafe_allow_html=True)
        st.markdown(f'<h1>{results[0][0]}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="subtitle">Confidence: {results[0][1]:.1f}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Running deeper analysis..."):
        analysis = get_plant_analysis(
            image=img,
            species_name=results[0][0],
            detected_disease=predicted_disease
        )

    st.markdown("### Detailed Analysis")
    st.markdown(analysis)
 
    st.markdown("### Other possible matches")
    for species, proba in results[1:]:
        st.write(species)
        st.progress(min(int(proba), 100))

    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].append({
        "image": uploaded_file,
        "species": results[0][0],
        "confidence": results[0][1],
        "disease": predicted_disease
    })
    
else:
    st.info("Upload a photo above to get started.")

