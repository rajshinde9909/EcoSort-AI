import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import tempfile
import os
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ------------------ Load Model ------------------
MODEL_PATH = "ecosortai_final_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------ Labels & Facts ------------------
CLASS_LABELS = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'Other Trash', 'white-glass']

WASTE_INFO = {
    "battery": {
        "description": "Batteries contain heavy metals that pollute soil and water.",
        "recycle": "Return to e-waste collection centers.",
        "hazard": "⚠️ High (toxic chemicals)",
        "time": "100+ years",
        "carbon_saving": "8 kg/kg",
        "landfill_reduction": "0.5 m³/ton",
        "tip": "Tape battery terminals before disposal."
    },
    "biological": {
        "description": "Organic waste such as food scraps and garden waste.",
        "recycle": "Compost at home or community bins.",
        "hazard": "✅ Low",
        "time": "2–12 weeks",
        "carbon_saving": "0.2 kg/kg",
        "landfill_reduction": "0.8 m³/ton",
        "tip": "Mix green and brown waste for efficient composting."
    },
    "plastic": {
        "description": "Plastics are non-biodegradable and harmful to oceans.",
        "recycle": "Send to authorized recycling plants.",
        "hazard": "⚠️ Medium-High",
        "time": "500–1000 years",
        "carbon_saving": "2.5 kg/kg",
        "landfill_reduction": "1.2 m³/ton",
        "tip": "Prefer reusable alternatives over single-use plastic."
    },
    "cardboard": {
        "description": "Recyclable paper-based material used in packaging.",
        "recycle": "Flatten and place in paper recycling bins.",
        "hazard": "✅ Low",
        "time": "2 months",
        "carbon_saving": "1.5 kg/kg",
        "landfill_reduction": "0.9 m³/ton",
        "tip": "Keep cardboard dry to ensure recyclability."
    },
    "clothes": {
        "description": "Textiles that can often be reused or recycled.",
        "recycle": "Donate wearable clothes, recycle unusable fabric.",
        "hazard": "⚠️ Medium (dyes, microfibers)",
        "time": "1–5 years",
        "carbon_saving": "3 kg/kg",
        "landfill_reduction": "0.6 m³/ton",
        "tip": "Repurpose old clothes into cleaning rags."
    },
    "brown-glass": {
        "description": "Brown-tinted glass, often used in bottles.",
        "recycle": "Sort by color and recycle at glass facilities.",
        "hazard": "⚠️ Medium (sharp edges)",
        "time": "❌ Never (doesn’t decompose)",
        "carbon_saving": "0.6 kg/kg",
        "landfill_reduction": "1.3 m³/ton",
        "tip": "Rinse bottles before recycling."
    },
    "green-glass": {
        "description": "Green-colored glass containers.",
        "recycle": "Sort by color for recycling.",
        "hazard": "⚠️ Medium (sharp edges)",
        "time": "❌ Never",
        "carbon_saving": "0.6 kg/kg",
        "landfill_reduction": "1.3 m³/ton",
        "tip": "Avoid mixing with other glass colors."
    },
    "white-glass": {
        "description": "Clear glass containers and jars.",
        "recycle": "Recycle separately from colored glass.",
        "hazard": "⚠️ Medium (sharp edges)",
        "time": "❌ Never",
        "carbon_saving": "0.6 kg/kg",
        "landfill_reduction": "1.3 m³/ton",
        "tip": "Remove caps and lids before recycling."
    },
    "metal": {
        "description": "Includes aluminum cans and steel products.",
        "recycle": "Rinse and recycle through metal facilities.",
        "hazard": "⚠️ Medium (sharp edges, rust)",
        "time": "50–200 years",
        "carbon_saving": "9 kg/kg",
        "landfill_reduction": "1.4 m³/ton",
        "tip": "Crush cans to save space."
    },
    "paper": {
        "description": "Common recyclable material made from wood pulp.",
        "recycle": "Recycle clean, dry paper.",
        "hazard": "✅ Low",
        "time": "2–6 weeks",
        "carbon_saving": "1.2 kg/kg",
        "landfill_reduction": "1.1 m³/ton",
        "tip": "Avoid recycling greasy or wet paper."
    },
    "shoes": {
        "description": "Footwear made from mixed materials (rubber, leather, fabric).",
        "recycle": "Donate wearable shoes, recycle rubber soles.",
        "hazard": "⚠️ Medium (synthetics take long to decompose)",
        "time": "25–40 years",
        "carbon_saving": "2 kg/kg",
        "landfill_reduction": "0.4 m³/ton",
        "tip": "Reuse old shoes for gardening or outdoor activities."
    },
    "trash": {
        "description": "General waste that cannot be recycled.",
        "recycle": "Dispose of in municipal solid waste bins.",
        "hazard": "⚠️ High (landfill pollution)",
        "time": "Varies (decades to never)",
        "carbon_saving": "0 (non-recyclable)",
        "landfill_reduction": "❌ None",
        "tip": "Reduce trash by practicing 3Rs: Reduce, Reuse, Recycle."
    }
}


RECYCLABILITY_SCORE = {
    "battery": 10, "biological": 90, "brown-glass": 85, "cardboard": 95, "clothes": 60,
    "green-glass": 85, "metal": 98, "paper": 96, "plastic": 50, "shoes": 40,
    "Other Trash": 10, "white-glass": 85
}

DID_YOU_KNOW = [
    "Recycling 1 aluminum can saves enough energy to run a TV for 3 hours 📺",
    "Plastic bottles can take up to 1000 years to decompose ⏳",
    "Composting food waste reduces methane emissions from landfills 🌱",
    "Recycling 1 ton of paper saves 17 trees 🌳",
    "Glass is 100% recyclable and can be reused endlessly without loss of quality 🍾"
]

# ------------------ Utility Functions ------------------
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def create_confidence_plot(preds, labels):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, preds * 100, color="#2ECC71")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Confidence %")
    plt.title("Confidence Distribution")
    st.pyplot(fig)

def recyclability_chart(score):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie([score, 100-score], labels=[f"{score}% Recyclable", ""],
           colors=["#27AE60", "#E5E8E8"], startangle=90, counterclock=False,
           wedgeprops={'width':0.3, 'edgecolor':'white'})
    st.pyplot(fig)

def generate_pdf(class_name, confidence, uploaded_image):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    W, H = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H - 50, "EcoSortAI - Waste Classification Report")

    # Metadata
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 70, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(40, H - 85, f"Model: {os.path.basename(MODEL_PATH)}")

    # Prediction
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, H - 120, f"Predicted Class: {class_name.upper()} ({confidence:.2f}%)")

    info = WASTE_INFO.get(class_name, {})
    lines = [
        f"Description: {info.get('description','N/A')}",
        f"Recycle: {info.get('recycle','N/A')}",
        f"Hazard: {info.get('hazard','N/A')}",
        f"Decomposition Time: {info.get('time','N/A')}",
        f"Eco Tip: {info.get('tip','N/A')}",
        f"Recyclability Score: {RECYCLABILITY_SCORE.get(class_name,'N/A')}/100",
        f"Carbon saving: {info.get('carbon_saving','N/A')}",
        f"Landfill reduction: {info.get('landfill_reduction','N/A')}"
    ]

    y = H - 140
    for line in lines:
        c.drawString(40, y, line)
        y -= 16

    # Insert image
    if uploaded_image is not None:
        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        uploaded_image.save(img_path)
        c.drawImage(img_path, 360, H - 320, width=200, height=200, preserveAspectRatio=True)
        os.remove(img_path)

    c.showPage()
    c.save()
    return tmp_file.name

# ------------------ Streamlit Layout ------------------
st.set_page_config(page_title="EcoSortAI", page_icon="♻️", layout="wide")

with st.sidebar:
    st.image(r"C:\Users\rushi\Downloads\EcoSortAI logo.jpg", width=80)
    st.title("EcoSortAI")
    choice = st.radio("Navigate", ["🏠 Home", "📤 Upload Image", "ℹ️ About Project", "📩 Contact Us"])
    st.markdown("---")
    st.info(random.choice(DID_YOU_KNOW))

if choice == "🏠 Home":
    st.title("♻️ Welcome to EcoSortAI")
    st.subheader("AI-powered Waste Classification & Recycling Assistant")
    st.write("""
    EcoSortAI helps you identify different types of waste and provides recycling 
    guidelines, environmental impact details, and disposal tips.
    """)

elif choice == "📤 Upload Image":
    st.header("Upload Waste Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        arr = preprocess_image(image)
        preds = model.predict(arr)[0]
        class_idx = np.argmax(preds)
        class_name = CLASS_LABELS[class_idx]
        confidence = preds[class_idx] * 100

        st.subheader(f"🔎 Predicted: **{class_name.upper()}** ({confidence:.2f}%)")
        st.progress(int(confidence))

        create_confidence_plot(preds, CLASS_LABELS)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📖 Description")
            st.write(WASTE_INFO.get(class_name, {}).get("description", "N/A"))

            st.markdown("### ♻️ How to Recycle")
            st.write(WASTE_INFO.get(class_name, {}).get("recycle", "N/A"))

            st.markdown("### ⚠️ Hazard Level")
            st.write(WASTE_INFO.get(class_name, {}).get("hazard", "N/A"))

        with col2:
            st.markdown("### ⏳ Decomposition Time")
            st.write(WASTE_INFO.get(class_name, {}).get("time", "N/A"))

            st.markdown("### 💡 Eco Tip")
            st.write(WASTE_INFO.get(class_name, {}).get("tip", "N/A"))

            st.markdown("### 🌍 Recyclability")
            recyclability_chart(RECYCLABILITY_SCORE.get(class_name, 50))

        # PDF download
        pdf_path = generate_pdf(class_name, confidence, image)
        with open(pdf_path, "rb") as f:
            st.download_button("💾 Download Report (PDF)", f, file_name="EcoSortAI_Report.pdf", mime="application/pdf")

elif choice == "ℹ️ About Project":
    st.header("About EcoSortAI")
    st.write("""
    EcoSortAI is a deep learning project that classifies waste into categories such as 
    plastic, glass, paper, and more.  
    It provides real-world recycling tips, eco-friendly disposal methods, and generates detailed PDF reports.
    """)

elif choice == "📩 Contact Us":
    st.header("Contact the Developers")
    st.write("📧 Email: rushikadam1912@gmail.com") 
    st.write("🌐 GitHub: [EcoSortAI Repository](https://github.com/)")
    st.write("💼 LinkedIn: EcoSortAI Team")
