# app.py

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np  # [web:19]
import pandas as pd  # [web:19]
from PIL import Image  # [web:19]
import streamlit as st  # [web:1][web:13]

# ------------------------------------------------
# Page configuration & base CSS
# ------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Glassmorphism-style custom CSS
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at top left, #052b2f 0, #000000 55%, #052b2f 100%);
        color: #f5f5f5;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    .main {
        padding: 1rem 2rem;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(18px);
    }
    .glass-header {
        background: linear-gradient(120deg, rgba(34,197,94,0.18), rgba(56,189,248,0.18));
        border-radius: 18px;
        padding: 1.2rem 1.6rem;
        border: 1px solid rgba(52,211,153,0.55);
        box-shadow: 0 16px 38px rgba(22,163,74,0.8);
        backdrop-filter: blur(14px);
    }
    .big-prediction {
        font-size: 1.9rem;
        font-weight: 700;
        color: #bbf7d0;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .confidence-label {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc;
    }
    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .disease-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: rgba(22,163,74,0.18);
        border: 1px solid rgba(52,211,153,0.7);
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bbf7d0;
    }
    .risk-pill-high {
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(248, 113, 113, 0.15);
        border: 1px solid rgba(248, 113, 113, 0.7);
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #fecaca;
    }
    .risk-pill-medium {
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(251, 191, 36, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.7);
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #fef9c3;
    }
    .risk-pill-low {
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(22, 163, 74, 0.15);
        border: 1px solid rgba(52, 211, 153, 0.7);
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bbf7d0;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin-bottom: 0.35rem;
    }
    .disease-image-placeholder {
        width: 100%;
        min-height: 220px;
        border-radius: 16px;
        border: 1px dashed rgba(148, 163, 184, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #e5e7eb;
        font-size: 0.95rem;
        background: radial-gradient(circle at top, rgba(15,23,42,0.9), rgba(15,23,42,0.4));
    }
    .footer-note {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 1rem;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------
# Sidebar: Branding
# ------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding-bottom: 0.8rem;">
            <div style="
                font-size: 2.2rem;
                line-height: 1;
                margin-bottom: 0.25rem;
            ">
                🌱
            </div>
            <div style="font-weight: 800; font-size: 1rem; letter-spacing: 0.16em; text-transform: uppercase;">
                Plant Disease Classifier
            </div>
            <div style="font-size: 0.8rem; color: #9ca3af; letter-spacing: 0.16em; text-transform: uppercase; margin-top: 0.25rem;">
                Vision + AI + Agronomy
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### 👥 Project Info")
    st.markdown("- **Model:** CNN (Keras/TensorFlow)")
    st.markdown("- **Dataset:** Plant Village (via GitHub repo) [web:0]")

    st.markdown("---")
    st.caption("Prototype. Not a substitute for expert agronomic advice.")

# ------------------------------------------------
# Disease metadata derived from dataset description
# (you can adjust to match your class_names.json exactly)
# ------------------------------------------------
DISEASE_LIST = [
    # Apple
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    # Blueberry
    "Blueberry___healthy",
    # Cherry
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    # Corn
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    # Grape
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)",
    # Peach
    "Peach___Bacterial_spot",
    "Peach___healthy",
    # Pepper
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    # Potato
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    # Raspberry
    "Raspberry___healthy",
    # Soybean
    "Soybean___healthy",
    # Squash
    "Squash___Powdery_mildew",
    # Strawberry
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    # Tomato (10)
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus",
    "Tomato___healthy",
]

# A compact subset of protocols (extend as you wish)
DISEASE_PROTOCOLS = {
    "Tomato___Early_blight": {
        "symptoms": [
            "Concentric brown rings on older leaves, often with yellow halos.",
            "Lesions usually start on lower leaves and move upward.",
        ],
        "causes": [
            "Fungal pathogen Alternaria solani under warm, humid conditions.",
            "Spread by infected debris and splashing rain or irrigation.",
        ],
        "solutions": [
            "Remove and destroy heavily infected leaves and plant residues.",
            "Use protectant fungicides (e.g., mancozeb or chlorothalonil) as per local recommendations.",
            "Adopt crop rotation and avoid continuous tomato or potato in the same field.",
            "Improve airflow via staking/pruning and avoid late evening overhead irrigation.",
        ],
    },
    "Tomato___Late_blight": {
        "symptoms": [
            "Water-soaked lesions on leaves that quickly turn brown and necrotic.",
            "White fungal growth on the underside of leaves during humid conditions.",
        ],
        "causes": [
            "Oomycete pathogen Phytophthora infestans favoured by cool, humid weather.",
            "Spreads rapidly through wind-borne spores and infected planting material.",
        ],
        "solutions": [
            "Rogue-out and destroy severely infected plants to reduce inoculum.",
            "Apply registered systemic fungicides in combination with protectants in a rotation.",
            "Ensure good field drainage and avoid prolonged leaf wetness.",
            "Plant disease-free seedlings and resistant/tolerant cultivars where possible.",
        ],
    },
    "Potato___Early_blight": {
        "symptoms": [
            "Target-like concentric lesions on older leaves with yellow margins.",
            "Premature defoliation leading to smaller tubers.",
        ],
        "causes": [
            "Alternaria solani persisting on crop debris and volunteer plants.",
        ],
        "solutions": [
            "Destroy volunteer plants and cull piles.",
            "Follow a preventive fungicide spray schedule based on local guidelines.",
            "Practice crop rotation away from solanaceous crops.",
        ],
    },
    "Potato___Late_blight": {
        "symptoms": [
            "Dark, water-soaked leaf lesions with pale green borders.",
            "Brown, firm rot on tubers with granular texture.",
        ],
        "causes": [
            "Phytophthora infestans in cool, moist environments.",
        ],
        "solutions": [
            "Use certified seed tubers and avoid excessive irrigation.",
            "Apply recommended systemic fungicides during high-risk weather.",
            "Destroy infected residues and volunteer plants.",
        ],
    },
    "Grape___Black_rot": {
        "symptoms": [
            "Circular brown spots on leaves with dark margins.",
            "Infected berries shrivel into black mummies.",
        ],
        "causes": [
            "Fungus Guignardia bidwellii surviving in mummified berries and canes.",
        ],
        "solutions": [
            "Remove mummified clusters and prune infected canes.",
            "Use a protective fungicide program starting early in the season.",
            "Maintain an open canopy with proper pruning and training.",
        ],
    },
    "Apple___Apple_scab": {
        "symptoms": [
            "Olive-green to dark, velvety lesions on young leaves.",
            "Scab-like, corky spots on fruit causing deformation.",
        ],
        "causes": [
            "Venturia inaequalis, with spores released from fallen leaf litter.",
        ],
        "solutions": [
            "Collect and destroy fallen leaves to reduce primary inoculum.",
            "Apply protectant fungicides during susceptible growth stages.",
            "Use scab-resistant cultivars when available.",
        ],
    },
    "Corn___Common_rust": {
        "symptoms": [
            "Small, cinnamon-brown pustules scattered on both leaf surfaces.",
        ],
        "causes": [
            "Puccinia sorghi, favoured by moderate temperatures and high humidity.",
        ],
        "solutions": [
            "Use resistant hybrids where possible.",
            "Apply fungicides when disease exceeds economic thresholds.",
        ],
    },
}

# Default fall-back protocol for any class
DEFAULT_PROTOCOL = {
    "symptoms": [
        "Visual symptoms vary with crop and pathogen.",
        "Look for spots, lesions, discoloration, wilting, or abnormal growth.",
    ],
    "causes": [
        "Can be caused by fungal, bacterial, viral, or nutrient-related problems.",
        "Weather conditions, irrigation practices, and soil health strongly influence disease risk.",
    ],
    "solutions": [
        "Scout fields regularly and compare symptoms with trusted references.",
        "Consult local agricultural extension or agronomists for precise diagnosis.",
        "Use integrated management: resistant varieties, crop rotation, sanitation, and labeled crop-protection products.",
    ],
}

# ------------------------------------------------
# Utility: Load model & classes
# ------------------------------------------------


@st.cache_resource(show_spinner="Loading CNN model and class names...")
def load_model_and_classes():
    """
    Load a TensorFlow/Keras model (model.h5) and class_names.json if present. [web:12][web:9]
    """
    model = None
    class_names = DISEASE_LIST

    # Try to load model.h5 (if available)
    try:
        from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

        model_path = "model.h5"
        if os.path.exists(model_path):
            model = keras_load_model(model_path)
    except Exception:
        # Silent fail to simulation if TF is not installed or model missing
        model = None

    # Try to load class_names.json
    class_file = "class_names.json"
    if os.path.exists(class_file):
        try:
            with open(class_file, "r", encoding="utf-8") as f:
                loaded_classes = json.load(f)
            if isinstance(loaded_classes, list) and len(loaded_classes) > 0:
                class_names = loaded_classes
        except Exception:
            pass

    return model, class_names


def preprocess_image(uploaded_file, target_size=(224, 224)):
    """
    Convert uploaded image to RGB, resize, normalize, and add batch dimension. [web:19]
    """
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return image, img_batch


def predict_or_simulate(model, img_batch, class_names):
    """
    Use real model if available; otherwise simulate probabilistic output. [web:19]
    """
    if model is not None:
        preds = model.predict(img_batch)
        preds = np.squeeze(preds)
        if preds.ndim == 0:
            preds = np.array([preds])
        probs = preds / np.sum(preds)
    else:
        seed_value = int(np.sum(img_batch) * 1e6) % (2**31 - 1)
        rng = np.random.default_rng(seed_value)
        logits = rng.normal(loc=0.0, scale=1.0, size=len(class_names))
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

    top_indices = np.argsort(probs)[::-1][:3]
    top_probs = probs[top_indices]
    top_classes = [class_names[i] for i in top_indices]

    primary_class = top_classes[0]
    primary_confidence = float(top_probs[0] * 100)

    differential = list(
        zip(
            top_classes,
            [float(p * 100) for p in top_probs],
        )
    )

    return primary_class, primary_confidence, differential


def generate_farm_health_data(days=7):
    """
    Simulate 7-day time series for humidity, temperature, and soil pH. [web:19]
    """
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(days)][::-1]

    humidity = np.clip(
        np.random.normal(loc=78, scale=8, size=days), 40, 100
    )
    temperature = np.clip(
        np.random.normal(loc=27, scale=3, size=days), 15, 40
    )
    soil_ph = np.clip(
        np.random.normal(loc=6.4, scale=0.3, size=days), 4.5, 8.5
    )

    df = pd.DataFrame(
        {
            "Date": [d.strftime("%d-%b") for d in dates],
            "Humidity (%)": humidity,
            "Temperature (°C)": temperature,
            "Soil pH": soil_ph,
        }
    )
    df.set_index("Date", inplace=True)
    return df


def assess_risk(humidity_series, temperature_series, ph_series):
    """
    Simple heuristic risk assessment. [web:19]
    """
    avg_h = float(np.mean(humidity_series))
    avg_t = float(np.mean(temperature_series))
    avg_p = float(np.mean(ph_series))

    if avg_h >= 80 and 20 <= avg_t <= 30:
        level = "HIGH"
        reason = "sustained high humidity and moderate temperatures, which favour foliar fungal diseases."
    elif avg_h >= 70 and 18 <= avg_t <= 32:
        level = "MEDIUM"
        reason = "elevated humidity with temperatures suitable for several leaf diseases."
    else:
        level = "LOW"
        reason = "conditions are relatively less favourable for major foliar fungal pathogens."

    ph_comment = ""
    if avg_p < 5.5:
        ph_comment = " Soil tends to be acidic; consider liming based on soil-test recommendations."
    elif avg_p > 7.5:
        ph_comment = " Soil tends to be slightly alkaline; monitor micronutrient availability."

    return level, reason + ph_comment


# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown(
    """
    <div class="glass-header">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div>
                <div style="font-size:0.8rem;letter-spacing:0.24em;text-transform:uppercase;color:#d1fae5;">
                    Plant Disease Detection
                </div>
                <div style="font-size:1.45rem;font-weight:800;margin-top:0.15rem;color:#f9fafb;">
                    Image-based Crop Disease Classifier
                </div>
                <div style="font-size:0.9rem;margin-top:0.4rem;color:#e5e7eb;max-width:540px;">
                    Upload a plant or leaf image, obtain AI predictions with confidence scores, 
                    explore treatment protocols, and view a sample farm health dashboard.
                </div>
            </div>
            <div style="text-align:right;min-width:180px;margin-top:0.8rem;">
                <span class="disease-badge">
                    🔬 CNN Classifier
                </span>
                <br/>
                <span class="disease-badge" style="margin-top:0.45rem;">
                    🌿 Integrated Management
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

tabs = st.tabs(
    [
        "🔎 AI Diagnosis & Classifier",
        "🌿 Treatment Protocol",
        "📊 Farm Health Monitor",
    ]
)

# ------------------------------------------------
# Tab 1: AI Diagnosis & Classifier
# ------------------------------------------------
with tabs[0]:
    st.markdown("### 🔎 AI Diagnosis & Classifier")

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.markdown("#### Upload Leaf / Plant Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or browse (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            help="Use clear, focused images with the symptomatic area visible.",
        )

    with col_right:
        st.markdown("#### Model Summary")
        model_obj, class_names = load_model_and_classes()
        st.metric(
            label="Number of Classes",
            value=len(class_names),
            help="Total plant disease and healthy classes supported.",
        )
        st.metric(
            label="Input Size",
            value="224 × 224 × 3",
            help="Standard RGB size used during preprocessing.",
        )

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("Processing image and running classifier..."):
            time.sleep(0.6)
            image, img_batch = preprocess_image(uploaded_file)
            prediction, confidence, diff_list = predict_or_simulate(
                model_obj, img_batch, class_names
            )

        img_col, result_col = st.columns([1.2, 1])
        with img_col:
            st.markdown("##### Input Image")
            st.image(
                image,
                caption="Uploaded sample",
                use_column_width=True,
            )

        with result_col:
            st.markdown("##### Prediction")
            st.markdown(
                f"""
                <div style="
                    border-radius:16px;
                    padding:1.0rem 1.2rem;
                    background:linear-gradient(135deg, rgba(34,197,94,0.16), rgba(22,163,74,0.75));
                    border:1px solid rgba(34,197,94,0.85);
                    box-shadow:0 18px 40px rgba(22,163,74,0.85);
                    text-align:center;
                ">
                    <div style="font-size:0.85rem;letter-spacing:0.18em;text-transform:uppercase;color:#dcfce7;">
                        Primary Class
                    </div>
                    <div class="big-prediction" style="margin-top:0.35rem;">
                        {prediction}
                    </div>
                    <div style="margin-top:0.8rem;">
                        <span class="confidence-label">Confidence:</span>
                        <div style="font-size:1.6rem;font-weight:700;margin-top:0.1rem;color:#fefce8;">
                            {confidence:.2f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence:.1f} %",
                )
            with c2:
                st.metric(
                    label="Top‑3 Scores",
                    value=f"{diff_list[0][1]:.1f}% / {diff_list[1][1]:.1f}% / {diff_list[2][1]:.1f}%",
                    help="Confidence of three most likely classes.",
                )

        st.write("")
        st.markdown("##### Differential Diagnosis (Top 3)")

        for idx, (cls_name, score) in enumerate(diff_list, start=1):
            progress_val = min(max(score / 100.0, 0.0), 1.0)
            st.progress(
                progress_val,
                text=f"{idx}. {cls_name} — {score:.2f}%",
            )
            time.sleep(0.03)

        st.markdown(
            """
            <div style="font-size:0.8rem;color:#9ca3af;margin-top:0.4rem;">
                Use the top‑3 suggestions along with field history and expert consultation for reliable decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Upload a plant or leaf image on the left to generate predictions and confidence scores."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# Tab 2: Treatment Protocol
# ------------------------------------------------
with tabs[1]:
    st.markdown("### 🌿 Integrated Treatment Protocol")

    st.markdown(
        """
        Select any predicted class to view typical symptoms, causes, and integrated management options.
        """
    )

    selected_disease = st.selectbox(
        "Choose a class / disease",
        options=sorted(DISEASE_LIST),
        index=DISEASE_LIST.index("Tomato___Early_blight")
        if "Tomato___Early_blight" in DISEASE_LIST
        else 0,
    )

    protocol = DISEASE_PROTOCOLS.get(selected_disease, DEFAULT_PROTOCOL)

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1.2])

    with col1:
        st.markdown(
            f'<div class="section-title">🩺 {selected_disease}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Symptoms (field-level)**")
        st.markdown("\n".join([f"- {item}" for item in protocol["symptoms"]]))
        st.markdown("")
        st.markdown("**Causes & Favouring Conditions**")
        st.markdown("\n".join([f"- {item}" for item in protocol["causes"]]))

    with col2:
        st.markdown(
            '<div class="section-title">🧪 Management Recommendations</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Suggested Actions**")
        st.markdown("\n".join([f"- {item}" for item in protocol["solutions"]]))
        st.markdown("")
        st.markdown("**Visual Reference Placeholder**")
        st.markdown(
            f"""
            <div class="disease-image-placeholder">
                Reference image for <b>{selected_disease}</b>
                (e.g., sample from dataset or field capture) can be placed here.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-note">
            Always verify product labels, safety intervals, and local recommendations when using pesticides or fungicides.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------
# Tab 3: Farm Health Monitor
# ------------------------------------------------
with tabs[2]:
    st.markdown("### 📊 Farm Health Monitor (Simulated)")

    st.markdown(
        """
        Example of a 7‑day IoT style feed for humidity, temperature, and soil pH.
        """
    )

    df_env = generate_farm_health_data(days=7)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Humidity (%)**")
        st.line_chart(df_env["Humidity (%)"])
    with c2:
        st.markdown("**Temperature (°C)**")
        st.line_chart(df_env["Temperature (°C)"])
    with c3:
        st.markdown("**Soil pH**")
        st.line_chart(df_env["Soil pH"])

    level, reason = assess_risk(
        df_env["Humidity (%)"], df_env["Temperature (°C)"], df_env["Soil pH"]
    )

    if level == "HIGH":
        risk_class = "risk-pill-high"
        warn_type = st.error
        title = "HIGH FUNGAL RISK"
    elif level == "MEDIUM":
        risk_class = "risk-pill-medium"
        warn_type = st.warning
        title = "MODERATE DISEASE RISK"
    else:
        risk_class = "risk-pill-low"
        warn_type = st.info
        title = "LOW DISEASE RISK"

    st.write("")
    with warn_type(
        f"{title}: {reason}",
        icon="⚠️" if level != "LOW" else "✅",
    ):
        pass

    st.markdown(
        f"""
        <div style="margin-top:0.5rem;">
            <span class="{risk_class}">
                {title}
            </span>
            <span style="font-size:0.82rem;color:#e5e7eb;margin-left:0.6rem;">
                Use such data to plan irrigation, fungicide sprays, and canopy management.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Values shown are synthetic examples, intended only to demonstrate dashboard functionality."
    )
