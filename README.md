# 🔥 Forest Fire Detection & Early Warning System

An AI-powered environmental monitoring system that combines real-time NASA satellite active fire feeds with deep learning Vision Transformer (`ViT`) image classification to detect and monitor wildfires.

---

## 🌟 Key Features

- **Real-Time Satellite Map:** Live active fire detection data pulled from NASA FIRMS (VIIRS SNPP NRT feed).
- **Vision Transformer Classifier:** Fine-tuned `google/vit-base-patch16-224` image classification model achieving **98.75% accuracy**.
- **Interactive Web Interface:** Streamlit dashboard with performance metrics, map visualization, and instant image testing.
- **Production Containerization:** Optimized `Dockerfile` and `docker-compose.yml` for CPU inference and instant deployment on Render, Railway, AWS, or GCP Cloud Run.

---

## 🚀 Quick Start with Docker (Recommended)

Run the entire application in a self-contained container with a single command:

```bash
docker compose up --build
```

Then visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💻 Local Manual Setup

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the dashboard:**
   ```bash
   streamlit run main.py
   ```

---

## ☁ Cloud Deployment

Refer to [`DEPLOYMENT.md`](file:///d:/AIML/forest-fire-detection/DEPLOYMENT.md) for full instructions on deploying to:
- **Render** (via `render.yaml`)
- **Railway** (via `railway.json`)
- **Google Cloud Run**
- **AWS App Runner / ECS**

---

## 📊 Model Evaluation

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.75% |
| **Precision** | 98.78% |
| **Recall** | 98.75% |
