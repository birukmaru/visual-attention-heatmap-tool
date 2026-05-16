# 🔍 Visual Attention Heatmap Tool

## Human Gaze Prediction for UI/UX & Marketing

An AI-powered visual saliency analysis tool that predicts **where human eyes look first, second, and third** on any image, screenshot, ad creative, or UI design. Built with Python and Streamlit for designers, marketers, and UX researchers.

---

## ✨ Features

### Core Analysis
- **Multi-layer saliency prediction** using deep learning, OpenCV spectral residual, and custom multi-scale fusion
- **Ranked fixation points** (1st, 2nd, 3rd Look) with coordinates, attention scores, and bounding circles
- **Heatmap overlay** with adjustable opacity and multiple colormaps
- **Focus intensity map** with contour-based attention visualisation
- **Eye-tracking scanpath simulation** showing predicted gaze path

### Design & Marketing Insights
- **Visual Hierarchy Score** (0–100) based on focus concentration, clutter, and contrast
- **CTA Visibility Assessment** — measures attention in call-to-action regions
- **Banner Blindness Risk** detection
- **Distraction Score** — identifies visual noise competing for attention
- **Attention Zone Distribution** — navbar, hero, mid-section, footer breakdown

### Advanced Modes
- **🖥️ UI/UX Audit Mode** — navbar attention, hero section dominance, CTA discoverability
- **📢 Ad Creative Mode** — product prominence, face bias detection, text competition score
- **📦 Batch Analysis** — process multiple images simultaneously
- **🔄 A/B Design Comparison** — compare two designs side-by-side with attention metrics

### Export Options
- Download original image, heatmap overlay, annotated prediction image
- **Professional PDF report** with metrics, visualisations, and recommendations
- Saliency map and focus intensity map exports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

```bash
# 1. Clone or download the project
cd visual_attention_heatmap

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install PyTorch for deep learning saliency
pip install torch torchvision
```

### Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🏗️ Project Structure

```
visual_attention_heatmap/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── assets/
│   └── style.css             # Custom CSS (dark theme, cards, animations)
├── models/
│   └── __init__.py           # Pre-trained model placeholder
├── utils/
│   ├── __init__.py           # Package exports
│   ├── saliency.py           # Multi-layer saliency engine
│   ├── visualization.py      # Heatmap & annotation rendering
│   ├── insights.py           # Design & marketing insight generation
│   └── report_generator.py   # PDF report builder (ReportLab)
└── sample_images/            # Place sample images here
```

---

## 🧠 Model Architecture

The saliency engine uses a **layered approach** with graceful fallback:

| Tier | Model | Availability |
|------|-------|-------------|
| 1 | **Deep Learning** — PyTorch gradient-based neural saliency | Optional (requires PyTorch) |
| 2 | **OpenCV Spectral Residual** — Hou & Zhang (2007) | Requires opencv-contrib |
| 3 | **Multi-Scale Fusion** — Custom pipeline (always available) | Built-in |

### Multi-Scale Fusion Pipeline
The built-in saliency engine combines four complementary methods:
1. **Spectral Residual** — Frequency-domain analysis of visual surprise
2. **Colour Contrast** — CIE-Lab colour opponency from the mean
3. **Luminance Contrast** — Multi-scale centre-surround mechanism
4. **Edge Density** — Canny edge detection smoothed to density map

Results are fused via weighted averaging with optional **Gaussian centre bias** to simulate natural human tendency to look toward the image centre.

### Fixation Detection
- Local maxima extraction using `peak_local_max` (scikit-image)
- Non-maximum suppression with adaptive minimum distance
- Ranked by saliency intensity with normalised scoring

---

## 📸 Example Use Cases

1. **Landing Page Optimisation** — Upload your webpage screenshot to see if the CTA button captures enough attention
2. **Ad Creative Testing** — Check if the product or headline dominates over background elements
3. **Email Marketing** — Verify that the key message and call-to-action are in high-attention zones
4. **App UI Review** — Analyse where users will look first on your mobile or web app interface
5. **Social Media Graphics** — Ensure the main message isn't lost in visual clutter
6. **A/B Testing** — Compare two design variants to determine which has better visual hierarchy

---

## 📊 Metrics Explained

| Metric | Range | What It Measures |
|--------|-------|-----------------|
| Primary Focus Strength | 0–100% | How strongly the main element anchors attention |
| Distraction Score | 0–100% | Level of visual noise and competing elements |
| Visual Hierarchy | 0–100 | Overall quality of visual structure and flow |
| CTA Visibility | 0–100% | Attention directed to the call-to-action region |
| Attention Spread | 0–100% | How evenly distributed attention is across the image |

---

## 🛠️ Configuration

### Streamlit Theme
Edit `.streamlit/config.toml` to customise the colour scheme.

### Analysis Parameters
All parameters are adjustable via the sidebar:
- **Heatmap Opacity** — blend strength of the heat overlay
- **Number of Hotspots** — how many fixation points to detect
- **Colormap** — Jet, Inferno, Hot, Turbo, Magma, Plasma
- **Centre Bias** — toggle natural centre-looking tendency
- **Analysis Mode** — Auto, UI/UX, Ad Creative, General

---

## 🔮 Future Improvements

- [ ] Integration with DeepGaze IIE pre-trained weights
- [ ] Real-time webcam eye-tracking overlay
- [ ] GPU acceleration for batch processing
- [ ] REST API endpoint for CI/CD integration
- [ ] Temporal saliency for video content
- [ ] Accessibility contrast checker integration
- [ ] Multi-language UI support
- [ ] Cloud deployment (Streamlit Cloud / AWS)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📧 Contact

For questions, feedback, or collaboration inquiries, please open a GitHub issue.

---

> **Built with** ❤️ using Python, Streamlit, OpenCV, and modern computer vision techniques.
