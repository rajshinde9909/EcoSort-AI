# eco_sort_gui_full.py
import os
import io
import tempfile
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tensorflow as tf

# ------------------ Load Model ------------------
MODEL_PATH = "ecosortai_final_model.h5"  # change if needed
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    message = f"Could not load model at '{MODEL_PATH}'.\nMake sure it's present and valid.\n\nError: {e}"
    raise RuntimeError(message)

# ------------------ Labels & Facts ------------------
CLASS_LABELS = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                'green-glass', 'metal', 'paper', 'plastic', 'shoes', ' Other Trash', 'white-glass']

# Facts (comprehensive for examiners)
WASTE_INFO = {
    "battery": {
        "description": "Batteries contain heavy metals (lead, cadmium, mercury) and are harmful.",
        "recycle": "Return to e-waste / battery collection centers. Do not throw in household Other Trash.",
        "hazard": "High — toxic if leaked into soil/water.",
        "time": "100+ years",
        "carbon_saving_kg_per_kg": 8.0,
        "landfill_reduction_m3_per_ton": 0.5,
        "tip": "Tape terminals and store for safe transport to recycling center."
    },
    "biological": {
        "description": "Organic waste (food scraps, garden waste) that is biodegradable.",
        "recycle": "Compost or municipal organic waste collections / anaerobic digesters.",
        "hazard": "Low if composted correctly; avoid mixing with plastics.",
        "time": "2–12 weeks (compostable)",
        "carbon_saving_kg_per_kg": 0.2,
        "landfill_reduction_m3_per_ton": 0.8,
        "tip": "Keep moist and balanced (greens/browns) for efficient composting."
    },
    "brown-glass": {
        "description": "Brown (amber) glass bottles and jars used for beverages.",
        "recycle": "Rinse and drop at glass recycling points; separate colors if your local scheme requires it.",
        "hazard": "Low (physical hazard if broken).",
        "time": "Non-biodegradable (effectively infinite)",
        "carbon_saving_kg_per_kg": 0.5,
        "landfill_reduction_m3_per_ton": 0.6,
        "tip": "Keep clean; remove lids if required by local rules."
    },
    "cardboard": {
        "description": "Boxes, cartons and corrugated cardboard.",
        "recycle": "Flatten, remove contaminants, and recycle with paper/cardboard streams.",
        "hazard": "Low.",
        "time": "2 months (varies)",
        "carbon_saving_kg_per_kg": 1.2,
        "landfill_reduction_m3_per_ton": 3.0,
        "tip": "Break down boxes to save space and speed recycling."
    },
    "clothes": {
        "description": "Textiles and garments (natural and synthetic).",
        "recycle": "Donate wearable items; textile-recycling programs convert fibers to new products.",
        "hazard": "Moderate due to dyes and mixed materials.",
        "time": "Months to years",
        "carbon_saving_kg_per_kg": 2.0,
        "landfill_reduction_m3_per_ton": 2.5,
        "tip": "Donate or upcycle instead of discarding."
    },
    "green-glass": {
        "description": "Green glass bottles/jars.",
        "recycle": "Recycle at glass collection; can be infinitely recycled.",
        "hazard": "Low (if broken physical hazard).",
        "time": "Non-biodegradable",
        "carbon_saving_kg_per_kg": 0.5,
        "landfill_reduction_m3_per_ton": 0.6,
        "tip": "Rinse and separate if required."
    },
    "metal": {
        "description": "Cans, tins and scrap metals (aluminium, steel).",
        "recycle": "Very recyclable — take to scrap dealers or metal recycling bins.",
        "hazard": "Low, but sharp edges can cause injury.",
        "time": "50–200 years",
        "carbon_saving_kg_per_kg": 10.0,
        "landfill_reduction_m3_per_ton": 1.8,
        "tip": "Crush cans to save space and rinse leftover contents."
    },
    "paper": {
        "description": "Newspapers, office paper, magazines.",
        "recycle": "Recycle in paper stream; keep dry and free of food contamination.",
        "hazard": "Low.",
        "time": "2–6 weeks",
        "carbon_saving_kg_per_kg": 1.4,
        "landfill_reduction_m3_per_ton": 3.3,
        "tip": "Reuse sheets for notes or crafts before recycling."
    },
    "plastic": {
        "description": "Plastic packaging, bottles, containers (various resin codes).",
        "recycle": "Check local acceptance; PET & HDPE widely accepted; others vary.",
        "hazard": "High — microplastics and long-term persistence in environment.",
        "time": "100s to 1000s of years",
        "carbon_saving_kg_per_kg": 2.5,
        "landfill_reduction_m3_per_ton": 2.0,
        "tip": "Rinse and sort by type if possible; avoid single-use plastics."
    },
    "shoes": {
        "description": "Footwear (synthetic and natural materials).",
        "recycle": "Donate if wearable; certain programs recycle soles into surfaces/insulation.",
        "hazard": "Moderate — mixed materials hard to recycle.",
        "time": "25–40 years for synthetics",
        "carbon_saving_kg_per_kg": 0.8,
        "landfill_reduction_m3_per_ton": 1.1,
        "tip": "Repair or donate before recycling/disposal."
    },
    "Other Trash": {
        "description": "Non-recyclable residual waste (mixed contaminants).",
        "recycle": "Not recyclable. Reduce by source separation; follow municipal guidance.",
        "hazard": "Varies; can contain hazardous items.",
        "time": "Varies widely",
        "carbon_saving_kg_per_kg": 0.0,
        "landfill_reduction_m3_per_ton": 0.0,
        "tip": "Segregate recyclables and hazardous waste first."
    },
    "white-glass": {
        "description": "Clear/white glass bottles/jars.",
        "recycle": "Highly recyclable; separate by color if required.",
        "hazard": "Low (broken glass hazard).",
        "time": "Non-biodegradable",
        "carbon_saving_kg_per_kg": 0.5,
        "landfill_reduction_m3_per_ton": 0.6,
        "tip": "Rinse and recycle."
    }
}

# A simple recyclability score baseline per class (0..100)
RECYCLABILITY_SCORE = {
    "battery": 10, "biological": 90, "brown-glass": 85, "cardboard": 95, "clothes": 60,
    "green-glass": 85, "metal": 98, "paper": 96, "plastic": 50, "shoes": 40, "Other Trash": 10, "white-glass": 85
}

# ------------------ Utility: create chart ------------------
def create_confidence_figure(confidences, labels):
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=100)
    x = np.arange(len(labels))
    ax.bar(x, np.array(confidences) * 100, color="#16A085")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Confidence %")
    ax.set_title("Confidence distribution across classes")
    plt.tight_layout()
    return fig

# ------------------ GUI ------------------
class EcoSortGUI:
    def __init__(self, root):
        self.root = root
        root.title("EcoSortAI - Waste Classification & Recycling Guide")
        root.geometry("1220x820")
        root.configure(bg="#f3f6f7")

        # Header
        header = tk.Frame(root, bg="#148f77", height=70)
        header.pack(fill="x")
        tk.Label(header, text="♻️  EcoSortAI - Waste Classification & Recycling Guide",
                 font=("Arial Rounded MT Bold", 20), bg="#148f77", fg="white").pack(pady=14)

        # Main container
        main = tk.Frame(root, bg="#f3f6f7")
        main.pack(fill="both", expand=True, padx=12, pady=12)

        # Left card (upload + preview)
        left_card = tk.Frame(main, bg="#ecf0f1", bd=1, relief="solid", width=420)
        left_card.pack(side="left", fill="y", padx=(0, 10))

        self.upload_btn = tk.Button(left_card, text="📂 Upload Image", bg="#1ABC9C", fg="white",
                                    font=("Segoe UI", 12, "bold"), padx=12, pady=8, relief="flat",
                                    command=self.upload_image)
        self.upload_btn.pack(pady=14)

        # image placeholder frame with border
        self.preview_frame = tk.Frame(left_card, bg="white", bd=2, relief="groove", width=400, height=400)
        self.preview_frame.pack(padx=16, pady=6)
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame, text="No image loaded", bg="white")
        self.preview_label.pack(expand=True)

        # small quick stats (under preview)
        stats_frame = tk.Frame(left_card, bg="#ecf0f1")
        stats_frame.pack(fill="x", pady=8, padx=8)
        self.stat_label = tk.Label(stats_frame, text="No prediction yet", bg="#ecf0f1", anchor="w")
        self.stat_label.pack(fill="x")

        # Right card (scrollable dashboard)
        right_card_container = tk.Frame(main, bg="#f3f6f7")
        right_card_container.pack(side="left", fill="both", expand=True)

        # Canvas for scrolling
        canvas = tk.Canvas(right_card_container, bg="#f3f6f7", highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_card_container, orient="vertical", command=canvas.yview)
        self.scrollable = tk.Frame(canvas, bg="white")

        self.scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Dashboard contents (inside self.scrollable)
        dash_title = tk.Label(self.scrollable, text="📊 Prediction Dashboard",
                              font=("Segoe UI", 16, "bold"), bg="white")
        dash_title.pack(anchor="w", pady=(10, 6), padx=12)

        # Prediction header
        self.pred_header = tk.Label(self.scrollable, text="Upload an image to start", font=("Segoe UI", 14),
                                    bg="white", justify="left", anchor="w")
        self.pred_header.pack(fill="x", padx=12)

        # Progress bar
        self.conf_bar = ttk.Progressbar(self.scrollable, orient="horizontal", length=520, mode="determinate")
        self.conf_bar.pack(padx=12, pady=8)

        # Details card (scrollable text area)
        details_card = tk.Frame(self.scrollable, bg="#fafafa", bd=1, relief="solid")
        details_card.pack(fill="both", padx=12, pady=8)

        details_label = tk.Label(details_card, text="Details", font=("Segoe UI", 12, "bold"), bg="#fafafa")
        details_label.pack(anchor="w", padx=8, pady=(8, 0))

        self.details_text = ScrolledText(details_card, wrap="word", width=80, height=10, font=("Segoe UI", 11),
                                         bg="white", relief="flat")
        self.details_text.pack(padx=8, pady=8, fill="both", expand=True)

        # Extra badges and numbers row
        extra_frame = tk.Frame(self.scrollable, bg="white")
        extra_frame.pack(fill="x", padx=12, pady=6)

        self.recyc_score_label = tk.Label(extra_frame, text="Recyclability Score: N/A",
                                          bg="white", font=("Segoe UI", 11, "bold"))
        self.recyc_score_label.pack(side="left", padx=6)

        self.carbon_label = tk.Label(extra_frame, text="Carbon saving (kg/kg): N/A",
                                     bg="white", font=("Segoe UI", 11))
        self.carbon_label.pack(side="left", padx=18)

        self.landfill_label = tk.Label(extra_frame, text="Landfill reduction (m³/ton): N/A",
                                       bg="white", font=("Segoe UI", 11))
        self.landfill_label.pack(side="left", padx=18)

        # Confidence distribution chart area
        chart_frame = tk.Frame(self.scrollable, bg="white")
        chart_frame.pack(fill="both", padx=12, pady=10)
        self.chart_frame = chart_frame  # store to update later

        # Download button and timestamp
        bottom_frame = tk.Frame(self.scrollable, bg="white")
        bottom_frame.pack(fill="x", padx=12, pady=(6, 18))

        self.download_btn = tk.Button(bottom_frame, text="💾 Download Full Report (PDF)",
                                      bg="#E67E22", fg="white", font=("Segoe UI", 12, "bold"),
                                      relief="flat", padx=12, pady=8, command=self.download_pdf)
        self.download_btn.pack(side="left", padx=8)
        self.download_btn.bind("<Enter>", lambda e: self.download_btn.config(bg="#cf7117"))
        self.download_btn.bind("<Leave>", lambda e: self.download_btn.config(bg="#E67E22"))

        self.timestamp_label = tk.Label(bottom_frame, text="", bg="white", font=("Segoe UI", 9), anchor="e")
        self.timestamp_label.pack(side="right", padx=8)

        # State
        self.uploaded_path = None
        self.last_prediction = None  # tuple (class_name, confidence, preds_array)

    # -------- Upload & Predict ----------
    def upload_image(self):
        path = filedialog.askopenfilename(title="Select an image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.uploaded_path = path
        self._display_preview(path)
        self._predict_image(path)

    def _display_preview(self, path):
        # Resize and show in preview frame
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            # scale to fit 380 box keeping aspect ratio
            max_size = 380
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self.tk_preview = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.tk_preview, text="")
        except Exception as e:
            self.preview_label.config(text="Preview error")

    def _predict_image(self, path):
        try:
            # small visual update
            self.pred_header.config(text="Analyzing image...")
            self.conf_bar["value"] = 0
            self.details_text.delete("1.0", tk.END)
            self.root_update()

            # load image for model
            img = Image.open(path).convert("RGB").resize((224, 224))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr, verbose=0)[0]  # length 12
            class_idx = int(np.argmax(preds))
            class_name = CLASS_LABELS[class_idx]
            confidence = float(preds[class_idx]) * 100.0

            # update progress bar and header
            self.conf_bar["value"] = confidence
            self.pred_header.config(text=f"🔎 Predicted: {class_name.upper()} ({confidence:.2f}%)")

            # details
            info = WASTE_INFO.get(class_name, {})
            rec_score = RECYCLABILITY_SCORE.get(class_name, 0)
            carbon = info.get("carbon_saving_kg_per_kg", info.get("carbon_saving_kg_per_kg", 0.0))
            landfill = info.get("landfill_reduction_m3_per_ton", info.get("landfill_reduction_m3_per_ton", 0.0))

            details_text = (
                f"🗑️ Class: {class_name.upper()}\n"
                f"📊 Confidence: {confidence:.2f}%\n\n"
                f"📖 Description:\n{info.get('description', 'N/A')}\n\n"
                f"♻️ How to Recycle:\n{info.get('recycle', 'N/A')}\n\n"
                f"⚠️ Hazard Level:\n{info.get('hazard', 'N/A')}\n\n"
                f"⏳ Estimated Decomposition Time: {info.get('time', 'N/A')}\n\n"
                f"💡 Eco Tip: {info.get('tip', 'N/A')}\n\n"
                "Suggested Disposal / Action:\n"
                "- Segregate from other waste streams.\n"
                "- Take to local recycling / hazardous collection if applicable.\n"
            )

            # update UI elements
            self.details_text.delete("1.0", tk.END)
            self.details_text.insert(tk.END, details_text)
            self.recyc_score_label.config(text=f"Recyclability Score: {rec_score}/100")
            self.carbon_label.config(text=f"Carbon saving (kg/kg): {carbon}")
            self.landfill_label.config(text=f"Landfill reduction (m³/ton): {landfill}")

            # confidence chart
            self._draw_confidence_chart(preds)

            # timestamp
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.timestamp_label.config(text=f"Last analyzed: {now}")

            self.last_prediction = (class_name, confidence, preds.copy())
        except Exception as e:
            messagebox.showerror("Prediction error", f"Failed to predict: {e}")

    def _draw_confidence_chart(self, preds):
        # clear previous
        for w in self.chart_frame.winfo_children():
            w.destroy()

        fig = create_confidence_figure(preds, CLASS_LABELS)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        plt.close(fig)

    def root_update(self):
        # small helper to update UI if needed
        self.root.update_idletasks()

    # ---------- PDF Download ----------
    def download_pdf(self):
        if not self.last_prediction:
            messagebox.showwarning("No prediction", "Please upload and analyze an image first.")
            return

        out_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                filetypes=[("PDF files", "*.pdf")],
                                                title="Save prediction report as PDF")
        if not out_path:
            return

        class_name, confidence, preds = self.last_prediction

        # Prepare temporary image files
        tmp_files = []
        try:
            # save preview image to temp
            if self.uploaded_path:
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                img = Image.open(self.uploaded_path).convert("RGB")
                # resize to fit page
                img_for_pdf = img.resize((600, 600), Image.LANCZOS)
                img_for_pdf.save(tmp_img.name, format="JPEG")
                tmp_files.append(tmp_img.name)
                tmp_img.close()
            # save chart to temp
            fig = create_confidence_figure(preds, CLASS_LABELS)
            tmp_chart = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp_chart.name, dpi=150, bbox_inches="tight")
            tmp_files.append(tmp_chart.name)
            tmp_chart.close()
            plt.close(fig)

            # create PDF
            c = canvas.Canvas(out_path, pagesize=letter)
            W, H = letter

            # Header
            c.setFont("Helvetica-Bold", 18)
            c.drawString(40, H - 50, "EcoSortAI - Waste Classification Report")

            # Small metadata
            c.setFont("Helvetica", 10)
            c.drawString(40, H - 70, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(40, H - 85, f"Model: {os.path.basename(MODEL_PATH)}")

            # Prediction block
            y = H - 110
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, f"Predicted Class: {class_name.upper()} ({confidence:.2f}%)")
            y -= 18
            c.setFont("Helvetica", 11)
            info = WASTE_INFO.get(class_name, {})
            lines = [
                f"Description: {info.get('description','N/A')}",
                f"How to Recycle: {info.get('recycle','N/A')}",
                f"Hazard: {info.get('hazard','N/A')}",
                f"Estimated Decomposition Time: {info.get('time','N/A')}",
                f"Eco Tip: {info.get('tip','N/A')}",
                f"Recyclability Score: {RECYCLABILITY_SCORE.get(class_name,'N/A')}/100",
                f"Carbon saving (kg/kg): {info.get('carbon_saving_kg_per_kg','N/A')}",
                f"Landfill reduction (m³/ton): {info.get('landfill_reduction_m3_per_ton','N/A')}"
            ]
            for line in lines:
                c.drawString(40, y, line)
                y -= 16

            # Insert uploaded image on the right of the text block
            if tmp_files:
                # uploaded image first
                if self.uploaded_path:
                    img_x = 360
                    img_y = H - 200
                    c.drawImage(tmp_files[0], img_x, img_y - 300 + 70, width=200, height=200, preserveAspectRatio=True)

                # add chart below text or to the right depending on space
                chart_x = 360
                chart_y = y - 40
                c.drawImage(tmp_files[-1], 40, chart_y - 220, width=500, height=220, preserveAspectRatio=True)

            c.showPage()
            c.save()
            messagebox.showinfo("Saved", f"Report saved to {out_path}")
        finally:
            # cleanup temp files
            for f in tmp_files:
                try:
                    os.remove(f)
                except:
                    pass

# ------------------ Run App ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EcoSortGUI(root)
    root.mainloop()
