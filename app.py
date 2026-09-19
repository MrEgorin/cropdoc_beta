"""CropDoctor - local version. Run:  python app.py   ->  http://127.0.0.1:5000"""
import base64, glob, io, json, os, re, sys, threading, webbrowser
from collections import defaultdict

try:
    import numpy as np
    import torch, torch.nn as nn, torch.nn.functional as F, timm
    from flask import Flask, jsonify, request, send_file
    from PIL import Image, ImageOps
except ImportError as e:
    sys.exit(f"\n[CropDoctor] Missing package: {e.name}\n"
             "Install everything with:  pip install -r requirements.txt\n")

try:  # optional: iPhone HEIC/HEIF support
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = [os.environ.get("MODEL_DIR", ""), HERE, os.path.join(HERE, "model"), os.path.dirname(HERE), os.getcwd()]


def find(*patterns):
    for folder in filter(None, SEARCH):
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(folder, pat)))
            if hits:
                return hits[0]
    sys.exit(f"\n[CropDoctor] Could not find {patterns[0]}.\n"
             f"Put your model files next to app.py or in the 'model' folder.\nSearched: {[s for s in SEARCH if s]}\n")


print("[CropDoctor] Loading model ...")
cfg = json.load(open(find("best_config*.json"), encoding="utf-8"))
classes = json.load(open(find("label_encoder*.json"), encoding="utf-8"))
SIZE, MEAN, STD = cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"]

# Same architecture as build_model() in the notebook
backbone = timm.create_model(cfg["config"]["backbone"], pretrained=False, num_classes=0)
with torch.no_grad():
    feat = backbone.eval()(torch.zeros(1, 3, SIZE, SIZE)).shape[1]
model = nn.Sequential(backbone, nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, cfg["num_classes"])))
weights_path = find("best_model*.pth")
try:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
except TypeError:  # very old torch without weights_only
    state = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state)
model.eval()
MEAN_T, STD_T = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
print("[CropDoctor] Model ready.")


# ---- merge duplicate labels (Bing/Google/Baidu copies, stray "()" suffixes) ----
def split_label(label):
    plant, disease = label.split(" __ ", 1)
    disease = re.sub(r"\s*\(\)\s*$", "", disease.strip())
    disease = re.sub(r"\s+(Baidu|Bing|Google)$", "", disease)
    return plant.strip(), disease


parts = [split_label(c) for c in classes]
by_plant = defaultdict(set)
for p, d in parts:
    by_plant[p].add(d)


def canonical(p, d):
    shorter = [o for o in by_plant[p] if d.startswith(o) and len(o) < len(d)]
    d = min(shorter, key=len) if shorter else d
    if d.lower().startswith(p.lower() + " "):
        d = d[len(p) + 1:]
    return p, d[:1].upper() + d[1:]


GROUPS = [canonical(p, d) for p, d in parts]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


@app.get("/")
def index():
    for path in (os.path.join(HERE, "static", "index.html"), os.path.join(HERE, "index.html")):
        if os.path.exists(path):
            return send_file(path)
    return "index.html not found next to app.py or in the 'static' folder.", 404


@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No file received."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
    except Exception:
        return jsonify(error="This file couldn't be read as an image. Try JPG, PNG, WebP, HEIC, TIFF or BMP."), 400

    prev = img.copy()
    prev.thumbnail((720, 720))
    buf = io.BytesIO()
    prev.save(buf, "JPEG", quality=85)

    x = img.resize((SIZE, SIZE), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1).float() / 255
    t = ((t - MEAN_T) / STD_T).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)[0].tolist()

    diseases, plants = defaultdict(float), defaultdict(float)
    for pr, (p, d) in zip(probs, GROUPS):
        diseases[(p, d)] += pr
        plants[p] += pr
    top = sorted(diseases.items(), key=lambda kv: -kv[1])[:5]
    plant, plant_p = max(plants.items(), key=lambda kv: kv[1])
    best = top[0][1]
    status = "confident" if best >= 0.6 else "uncertain" if best >= 0.25 else "unclear"
    return jsonify(
        status=status,
        plant={"name": plant, "prob": plant_p},
        diseases=[{"plant": p, "disease": d, "prob": pr} for (p, d), pr in top],
        preview="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"
    print(f"\n[CropDoctor] Open  {url}  in your browser (Ctrl+C to stop)\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    except OSError:
        sys.exit(f"\n[CropDoctor] Port {port} is busy. Try another:  set PORT=5001  (Windows)  /  PORT=5001 python app.py\n")
