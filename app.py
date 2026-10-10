from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
import sqlite3, qrcode, io, secrets, re, os
from werkzeug.utils import secure_filename

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
DB = ROOT / "medicines.db"
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BYTES = 8 * 1024 * 1024
MODEL_XML = ROOT / "model" / "medicine_model.xml"
LABELS_FILE = ROOT / "model" / "labels.txt"
compiled = None
model_labels = []
model_message = "Model not installed. Add and validate your trained OpenVINO model."

def db_conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS medicines(
          medicine_id TEXT PRIMARY KEY, name TEXT NOT NULL, batch_number TEXT NOT NULL,
          manufacturer TEXT NOT NULL, manufacture_date TEXT, expiry_date TEXT NOT NULL,
          image_class TEXT NOT NULL, created_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS scans(
          scan_id INTEGER PRIMARY KEY AUTOINCREMENT, medicine_id TEXT NOT NULL,
          scanned_at TEXT NOT NULL, result TEXT NOT NULL, image_label TEXT,
          FOREIGN KEY(medicine_id) REFERENCES medicines(medicine_id))""")

def init_model():
    global compiled, model_labels, model_message
    if not MODEL_XML.exists():
        return
    try:
        from openvino import Core
        core = Core()
        compiled = core.compile_model(str(MODEL_XML), "CPU")
        if LABELS_FILE.exists():
            model_labels = [s.strip() for s in LABELS_FILE.read_text(encoding="utf-8").splitlines() if s.strip()]
        model_message = "OpenVINO model loaded. Confirm preprocessing and labels match training."
    except Exception as e:
        model_message = f"OpenVINO load error: {e}"

def new_id():
    # Random package-level identifier; uniqueness is enforced by the SQLite primary key.
    return "MED-" + secrets.token_hex(8).upper()

def run_openvino(path):
    if compiled is None:
        return {"configured": False, "label": None, "confidence": None,
                "message": model_message}
    try:
        import cv2, numpy as np
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Image could not be decoded.")
        inp = compiled.input(0)
        shape = list(inp.shape)
        if len(shape) != 4 or shape[1] not in (1, 3):
            raise ValueError(f"Expected NCHW image model; actual input shape {shape}")
        h, w = int(shape[2]), int(shape[3])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h))
        arr = image.astype(np.float32) / 255.0
        if shape[1] == 1:
            arr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None] / 255.0
        arr = np.transpose(arr, (2,0,1))[None, ...]
        output = compiled({inp: arr})
        scores = np.asarray(next(iter(output.values()))).reshape(-1)
        ix = int(np.argmax(scores))
        return {"configured": True, "label": model_labels[ix] if ix < len(model_labels) else f"class_{ix}",
                "confidence": round(float(scores[ix]), 4),
                "message": "Classification result only; not proof of authenticity."}
    except Exception as e:
        return {"configured": True, "label": None, "confidence": None,
                "message": f"Inference failed: {e}"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({"database": "connected", "openvino_model_loaded": compiled is not None,
                    "openvino_message": model_message})

@app.route("/api/medicines", methods=["GET"])
def list_medicines():
    with db_conn() as c:
        rows = c.execute("SELECT medicine_id,name,batch_number,manufacturer,expiry_date FROM medicines ORDER BY created_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/medicines", methods=["POST"])
def register_medicine():
    # In a deployed system, protect this route with authenticated admin access.
    data = request.get_json(silent=True) or request.form
    name = str(data.get("name","")).strip()
    batch = str(data.get("batch_number","")).strip()
    manufacturer = str(data.get("manufacturer","")).strip()
    mfg = str(data.get("manufacture_date","")).strip()
    expiry = str(data.get("expiry_date","")).strip()
    image_class = str(data.get("image_class","")).strip() or name
    if not all([name,batch,manufacturer,expiry]):
        return jsonify({"message":"Name, batch number, manufacturer and expiry date are required."}),400
    if len(name)>150 or len(batch)>100 or len(manufacturer)>150:
        return jsonify({"message":"One or more fields are too long."}),400
    medicine_id = new_id()
    try:
        with db_conn() as c:
            c.execute("""INSERT INTO medicines
            (medicine_id,name,batch_number,manufacturer,manufacture_date,expiry_date,image_class,created_at)
            VALUES(?,?,?,?,?,?,?,?)""",
            (medicine_id,name,batch,manufacturer,mfg,expiry,image_class,datetime.now(timezone.utc).isoformat()))
    except sqlite3.IntegrityError:
        return jsonify({"message":"ID collision. Please retry."}),409
    return jsonify({"message":"Medicine registered. Print this QR and attach it to exactly one package.",
                    "medicine":{"medicine_id":medicine_id,"name":name,"batch_number":batch,
                    "manufacturer":manufacturer,"expiry_date":expiry,
                    "qr_url":f"/api/qr/{medicine_id}.png"}}),201

@app.route("/api/qr/<medicine_id>.png")
def qr_image(medicine_id):
    with db_conn() as c:
        row=c.execute("SELECT medicine_id FROM medicines WHERE medicine_id=?",(medicine_id,)).fetchone()
    if row is None:
        return jsonify({"message":"Medicine ID not found."}),404
    # QR encodes only the opaque unique ID, not personal/medical information.
    img=qrcode.make(row["medicine_id"])
    buf=io.BytesIO(); img.save(buf,format="PNG"); buf.seek(0)
    return send_file(buf,mimetype="image/png",download_name=f"{medicine_id}.png")

@app.route("/api/verify", methods=["POST"])
def verify():
    qr_data=(request.form.get("qr_data") or "").strip()
    image=request.files.get("image")
    if not qr_data: return jsonify({"message":"Scan or enter a QR ID."}),400
    if not image or not image.filename: return jsonify({"message":"Upload a package image."}),400
    ext=Path(secure_filename(image.filename)).suffix.lower()
    if ext not in ALLOWED: return jsonify({"message":"Allowed: JPG, JPEG, PNG, WEBP."}),400
    image.stream.seek(0,os.SEEK_END); size=image.stream.tell(); image.stream.seek(0)
    if size==0 or size>MAX_BYTES: return jsonify({"message":"Image must be non-empty and at most 8 MB."}),400
    # Accept raw MED-... IDs or QR URLs with ?id=... / ?medicine_id=...
    parsed=urlparse(qr_data)
    id_from_url=parse_qs(parsed.query).get("id",[None])[0] or parse_qs(parsed.query).get("medicine_id",[None])[0]
    medicine_id=(id_from_url or (parsed.fragment if parsed.scheme else qr_data)).strip()
    if not re.fullmatch(r"MED-[A-F0-9]{16}",medicine_id):
        return jsonify({"status":"NOT_REGISTERED","message":"QR content is not a valid system medicine ID.",
                        "medicine":{},"image_screening":{},"scan_count":0}),200
    filename=secrets.token_hex(12)+ext
    path=UPLOADS/filename
    image.save(path)
    screening=run_openvino(path)
    with db_conn() as c:
        medicine=c.execute("SELECT * FROM medicines WHERE medicine_id=?",(medicine_id,)).fetchone()
        if medicine is None:
            status="NOT_REGISTERED"; message="This identifier is not registered."
            count=0
        else:
            count=c.execute("SELECT COUNT(*) FROM scans WHERE medicine_id=?",(medicine_id,)).fetchone()[0]
            # repeated-use flag is a review signal; do not claim counterfeit based only on repeats
            if count>0:
                status="SUSPICIOUS"; message="This identifier has been scanned before. Investigate reuse and distribution history."
            else:
                status="REGISTERED"; message="Identifier found in the registered database. This alone does not prove the package is genuine."
            c.execute("INSERT INTO scans(medicine_id,scanned_at,result,image_label) VALUES(?,?,?,?)",
                      (medicine_id,datetime.now(timezone.utc).isoformat(),status,screening.get("label")))
            count+=1
        details={} if medicine is None else {k:medicine[k] for k in ("name","batch_number","manufacturer","expiry_date","image_class")}
    return jsonify({"status":status,"message":message,"medicine":details,
                    "image_screening":screening,"scan_count":count,
                    "note":"AI image classification and QR registration are supporting checks, not regulatory proof of authenticity."})

if __name__=="__main__":
    init_db()
    init_model()
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1",port=5000,debug=True)
