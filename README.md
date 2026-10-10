# Counterfeit Medicine Verification — project implementation

## Setup on Windows / PowerShell
From this folder in VS Code, run each command separately. This avoids PowerShell's blocked Activate.ps1 script:

```powershell
python -m venv openvino_env
.\openvino_env\Scripts\python.exe -m pip install --upgrade pip
.\openvino_env\Scripts\python.exe -m pip install -r requirements.txt
.\openvino_env\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000

## Functions
- Register a package record (unique random package ID is generated server-side).
- Generate/download QR PNG that encodes only the unique ID.
- SQLite stores medicine records and scan history.
- Scan QR in browser or manually enter the ID.
- Upload package image; Flask runs OpenVINO inference only if a compatible model is installed.
- Repeated scans are flagged for review; they are not automatically proof of counterfeiting.

## Dataset in your report
Mobile-Captured Pharmaceutical Medication Packages, Mendeley Data v1, DOI 10.17632/bjy2svvmn8.1: https://data.mendeley.com/datasets/bjy2svvmn8/1
Reported in the project: 3,900 images from 150 packages. Download the dataset from the publisher page, read its licence/terms, and keep the original train/validation/test split by package where possible to prevent near-duplicate leakage. The dataset contains package identity images, not ground-truth counterfeit labels. It can support package-class recognition; counterfeit classification requires additional properly labelled genuine/counterfeit examples.

## OpenVINO
Install is included in requirements. Check installation in another terminal with:
` .\openvino_env\Scripts\python.exe -c "from openvino import Core; print(Core().available_devices)" `
The trained model is NOT included. See model/README.txt. The app reports model status at http://127.0.0.1:5000/api/health.

## Final-project hardening still required
This is a working implementation baseline, but before calling it production-ready, add admin login/roles, CSRF protection for web forms, rate limiting, robust image content validation, retention/deletion rules for uploaded images, audit logs, deployment configuration, backup strategy, and security testing. Use approved records; do not use it for actual clinical decisions.
