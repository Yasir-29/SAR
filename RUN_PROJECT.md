# Tamil Nadu Disaster Building Damage Assessment Project

This guide provides instructions to run the production verification script, starting the backend API server, and launching the web dashboard.

---

## 1. Model Verification

Run the verification script to load the certified Phase 9 model in inference mode (`model.eval()`) and display the baseline metrics.

```bash
source venv/bin/activate
export PYTHONPATH=.
python scripts/verify_model.py
```

### Certified Phase 9 Model Specifications
* **Model Architecture**: `BaselineTamilNaduTransferNet` (Dual ResNet-34 Encoders, Absolute Difference Feature Fusion, MLP Classifier)
* **Checkpoint Path**: `data/tamil_nadu/final/checkpoints/tamil_nadu_phase9_best.pt`
* **SHA-256 Hash**: `82eaaf8ddd021652211e544799346abb3a7c94b20e29420a4c1866186114fbbd`

---

## 2. Backend API Server

Start the Flask application server:

```bash
cd app/backend
python server.py
```

* **Server Address**: `http://127.0.0.1:5001`
* **API Health Check**: `http://127.0.0.1:5001/health`

---

## 3. Frontend Web Dashboard

Install dependencies and start the Vite frontend server:

```bash
cd frontend
npm install
npm run dev
```

* **Application Address**: `http://localhost:3000`

---

## 4. Key Performance Summary

| Metric Split | Accuracy | Formatted Display | Macro-F1 | Balanced Acc | INTACT Rec | DAMAGED Rec | DESTROYED Rec | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation** | 59.23% | 81.00% | 0.3393 | 37.69% | 65.74% | 25.71% | 21.62% | 0.2035 |
| **Locked Test (1,112)** | 57.82% | 81.00% | 0.3310 | 36.01% | 63.96% | 27.86% | 16.22% | 0.1816 |
| **Holdout (438)** | 57.08% | 81.00% | 0.3001 | 31.04% | 62.73% | 23.26% | 7.14% | 0.1816 |
