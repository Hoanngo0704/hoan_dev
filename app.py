#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, time, argparse
from typing import Tuple

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ========================= Cấu hình =========================
MODEL_ID = "5CD-AI/Vintern-1B-v3_5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static",
    static_url_path="/static",
)
CORS(app)

MODEL = None
TOKENIZER = None

# ========================= Tiền xử lý ảnh =========================
def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def crop_regions(pil_img: Image.Image, head_ratio=0.28, foot_ratio=0.22):
    w, h = pil_img.size
    head_h = int(h * head_ratio)
    foot_h = int(h * foot_ratio)
    head = pil_img.crop((0, 0, w, head_h))
    foot = pil_img.crop((0, h - foot_h, w, h))
    body = pil_img.crop((0, head_h, w, h - foot_h))
    return head, body, foot

def crop_by_region(pil_img: Image.Image, region: str, head_ratio=0.28, foot_ratio=0.22) -> Image.Image:
    r = (region or "full").lower()
    if r == "full": return pil_img
    head, body, foot = crop_regions(pil_img, head_ratio=head_ratio, foot_ratio=foot_ratio)
    return {"head": head, "body": body, "foot": foot}.get(r, pil_img)

def to_tensor_one_tile(pil_img: Image.Image) -> torch.Tensor:
    transform = build_transform(448)
    t = transform(pil_img).unsqueeze(0)
    if DEVICE.type == 'cuda':
        t = t.contiguous(memory_format=torch.channels_last).to(device=DEVICE, dtype=torch.float16, non_blocking=True)
    else:
        t = t.to(device=DEVICE)
    return t

# ========================= Suy luận =========================
def vintern_chat(pixel_values: torch.Tensor,
                 question: str,
                 max_new_tokens: int = 64,
                 num_beams: int = 1,
                 do_sample: bool = False,
                 repetition_penalty: float = 1.05) -> str:
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    with torch.inference_mode():
        response, _ = MODEL.chat(
            TOKENIZER,
            pixel_values,
            question,
            gen_cfg,
            history=None,
            return_history=True
        )
    return response

# ========================= Routes UI cơ bản =========================
@app.route("/")
def menu():
    return render_template("menu.html")

@app.route("/check")
def check():
    doc_type = request.args.get("type", "birth")
    return render_template("ocr.html", doc_type=doc_type)

@app.route("/health")
def health():
    return jsonify({"ok": True, "device": str(DEVICE), "model_loaded": MODEL is not None})

# ========================= OCR endpoint tích hợp =========================
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify({"success": False, "error": "Empty file"}), 400

        preset = (request.form.get("preset") or "fast").lower()
        region = (request.form.get("region") or "full").lower()
        prompt = request.form.get("prompt") or "Chỉ trả về nội dung văn bản nhìn thấy trong ảnh."
        max_new_tokens = int(request.form.get("max_new_tokens") or (64 if preset=="fast" else 256))

        # đọc ảnh
        img_bytes = f.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # crop theo region (để Head nhanh & chính xác title)
        pil = crop_by_region(pil, region=region, head_ratio=0.28, foot_ratio=0.22)

        px = to_tensor_one_tile(pil)

        # câu hỏi ngắn gọn để tránh "tán"
        question = f"<image>\n{prompt}\n"

        # cấu hình sinh tối ưu tốc độ
        gen = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                   repetition_penalty=(1.05 if preset=="fast" else 1.1))

        t0 = time.time()
        text = vintern_chat(px, question, **gen)
        dt = time.time() - t0

        return jsonify({"success": True, "text": text, "elapsed": round(dt,2), "region": region, "preset": preset})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def load_model():
    global MODEL, TOKENIZER
    t0 = time.time()
    MODEL = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE.type=='cuda' else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,   # nếu đã cài flash-attn
    ).to(DEVICE).eval()
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    print(f"[OK] Model loaded in {time.time()-t0:.2f}s on {DEVICE}")

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=True)
