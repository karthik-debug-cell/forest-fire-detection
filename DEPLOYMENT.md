# 🚀 Forest Fire Detection - Deployment Guide

This project is fully containerized and configured for production deployment using **Docker**, compatible with any modern container platform: **Docker Compose**, **Render**, **Railway**, **Google Cloud Run**, and **AWS App Runner**.

---

## 📋 System Requirements & Architecture

- **Base Image:** `python:3.10-slim`
- **Framework:** Streamlit + PyTorch CPU + Hugging Face Transformers
- **Model:** Vision Transformer (`ViT-Base-Patch16-224`) fine-tuned for wildfire classification
- **Satellite Integration:** NASA FIRMS VIIRS Active Fire NRT Feed
- **Recommended Memory:** 2 GB minimum (3 GB recommended for inference)
- **Port:** `8501` (or dynamic via `$PORT` environment variable)

---

## 🐳 Option 1: Run with Docker (Local or VM / VPS)

### Method A: Using Docker Compose (Recommended)

1. Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine is installed and running.
2. In the project root directory, run:
   ```bash
   docker compose up --build
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```
4. To stop the container:
   ```bash
   docker compose down
   ```

### Method B: Using Plain Docker CLI

1. **Build the Docker Image:**
   ```bash
   docker build -t forest-fire-detection:latest .
   ```

2. **Run the Container:**
   ```bash
   docker run -d --name forest-fire-app -p 8501:8501 -e PORT=8501 -e NASA_FIRMS_API_KEY="f3fac5e638c9d20758c3ffb1b43c8e8b" forest-fire-detection:latest
   ```

3. **Check Logs:**
   ```bash
   docker logs -f forest-fire-app
   ```

4. **Stop Container:**
   ```bash
   docker stop forest-fire-app && docker rm forest-fire-app
   ```

---

## ☁ Option 2: Deploy to Render

1. Push your changes to GitHub:
   ```bash
   git add .
   git commit -m "Add Docker deployment configuration"
   git push origin main
   ```
2. Go to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** > **Web Service**.
4. Connect your GitHub repository `karthik-debug-cell/forest-fire-detection`.
5. Select **Docker** as the runtime (it automatically detects the `Dockerfile`).
6. Set:
   - **Instance Type:** Standard (at least 2GB RAM required for Vision Transformer inference)
   - **Environment Variables:**
     - `PORT`: `8501`
     - `NASA_FIRMS_API_KEY`: `f3fac5e638c9d20758c3ffb1b43c8e8b`
7. Click **Create Web Service**. Render builds the image and gives you a public HTTPS URL.

---

## 🚂 Option 3: Deploy to Railway

1. Push your repository to GitHub.
2. Log in to [Railway.app](https://railway.app/).
3. Click **New Project** > **Deploy from GitHub repo**.
4. Select `forest-fire-detection`.
5. Railway reads `railway.json` and builds from `Dockerfile` automatically.
6. In **Settings** > **Networking**, click **Generate Domain** to get a public URL.

---

## ☁ Option 4: Deploy to Google Cloud Run

1. Build and tag the image using Google Artifact Registry or Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT_ID]/forest-fire-detection
   ```
2. Deploy to Cloud Run with 2GB memory allocated:
   ```bash
   gcloud run deploy forest-fire-detection \
     --image gcr.io/[PROJECT_ID]/forest-fire-detection \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --port 8501
   ```

---

## 🔒 Environment Variables Reference

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PORT` | Listening port for Streamlit web server | `8501` |
| `NASA_FIRMS_API_KEY` | NASA FIRMS MAP API key for live satellite fire feeds | Included default key |

---

## 🩺 Health Check Endpoint

Streamlit provides an internal healthcheck endpoint at:
```
http://localhost:8501/_stcore/health
```
This is configured automatically in `Dockerfile` and `docker-compose.yml`.

