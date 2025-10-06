
# EcoSort-AI
Garbage Material Classification
=======
# ♻️ EcoSortAI – Smart Waste Classification & Recycling Guide

EcoSortAI is an AI-powered waste classification system that uses **Deep Learning** (CNN) to automatically identify different types of waste from images and provide **detailed recycling & disposal guidance**.  

The project includes:
- ✅ A trained **TensorFlow/Keras model**  
- ✅ An interactive **Tkinter desktop GUI**  
- ✅ A **Streamlit Web App** for deployment  
- ✅ Automatic **PDF report generation** with recycling details
  
---

## 🚀 Features

- 🖼 **Upload waste image** → Get instant classification  
- 📊 **Confidence distribution chart** for model predictions  
# ♻️ EcoSort-AI — Smart Waste Classification & Recycling Guide

EcoSort-AI is an AI-powered waste classification project that uses deep learning (CNN) to identify types of waste from images and provide recycling and disposal guidance.

The project in this repository includes:
- A TensorFlow / Keras model for classification
- A Tkinter desktop GUI (`GUI.py`)
- A Streamlit web app (`app.py`)
- PDF report generation and utility scripts

## Demo

Place screenshots in an `assets/` folder and reference them here (optional):

![Streamlit Demo](assets/streamlit_demo.png)
![GUI Demo](assets/gui_demo.png)

## Features

- Upload an image and get an automatic waste-class prediction
- Prediction confidence scores and simple visualization
- Recycling/disposal information per class
- Exportable PDF reports for each prediction

## Dataset

The dataset used to train models was split into train/validation/test and contains classes such as: Battery, Biological, Brown Glass, Cardboard, Clothes, Green Glass, Metal, Paper, Plastic, Shoes, Trash, White Glass.

## Tech Stack

- TensorFlow / Keras
- Tkinter (desktop GUI)
- Streamlit (web UI)
- ReportLab (PDF generation)
- Pandas, NumPy, Matplotlib

## Installation

Clone the repo and install dependencies:

```powershell
git clone https://github.com/rajshinde9909/EcoSort-AI.git
cd EcoSort-AI-main
pip install -r requirements.txt
```

Run the desktop GUI:

```powershell
python GUI.py
```

Run the Streamlit app:

```powershell
streamlit run app.py
```

## Results

Typical final model accuracy and training curves are included in the project artifacts (if present). Your mileage depends on dataset, training settings, and preprocessing.

## License

MIT License © 2025 Rushikesh Kadam

## Acknowledgements

- TensorFlow team
- Streamlit community
- Open-source datasets and contributors

If you want, I can also create an `assets/` folder and a curated `requirements.txt` tuned for Streamlit Cloud deployment.
🏙 Smart cities for automated waste sorting


