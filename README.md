# AI People Reader / Presentation Analysis (Streamlit)

This is a Streamlit app that:
- Uploads a presentation video
- Runs analysis (MediaPipe Pose + FaceMesh, with a fallback mode)
- Generates a **DOCX** report
- Optionally generates a dot-motion visualization MP4

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Render

This repo includes:
- `render.yaml` (Render service definition)
- `.streamlit/config.toml` (includes 1GB upload limits [[memory:10928783]])

### Option A: Deploy via Render Blueprint (recommended)
- Push this project to GitHub
- In Render: **New +** → **Blueprint**
- Select your GitHub repo
- Render will read `render.yaml` and deploy

### Option B: Manual Render Web Service
- In Render: **New +** → **Web Service**
- Connect your GitHub repo
- **Build command**:
  - `pip install -r requirements.txt`
- **Start command**:
  - `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Add env var:
  - `PYTHON_VERSION=3.10.13` [[memory:5590605]]


