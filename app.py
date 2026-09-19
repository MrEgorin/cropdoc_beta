import base64, glob, io, json, os, re
from collections import defaultdict

import torch, torch.nn as nn, torch.nn.functional as F, timm
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

try:  # optional: iPhone HEIC/HEIF support
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "model"))


def find(pattern):
    hits = sorted(glob.glob(os.path.join(MODEL_DIR, pattern)))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {MODEL_DIR}")
    return hits[0]


cfg = json.load(open(find("best_config*.json"), encoding="utf-8"))
classes = json.load(open(find("label_encoder*.json"), encoding="utf-8"))
SIZE, MEAN, STD = cfg["img_size"], cfg["imagenet_mean"], cfg["imagenet_std"]

# Same architecture as build_model() in the notebook: timm backbone + Dropout/Linear head
backbone = timm.create_model(cfg["config"]["backbone"], pretrained=False, num_classes=0)
with torch.no_grad():
    feat = backbone.eval()(torch.zeros(1, 3, SIZE, SIZE)).shape[1]
model = nn.Sequential(backbone, nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, cfg["num_classes"])))
model.load_state_dict(torch.load(find("best_model*.pth"), map_location="cpu"))
model.eval()

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

# ---- app ----
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


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
    t = torch.from_numpy(__import__("numpy").asarray(x)).permute(2, 0, 1).float() / 255
    t = ((t - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]).unsqueeze(0)
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
    app.run(host="127.0.0.1", port=5000)
