import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import time
import argparse
import sys
"""
url: https://huggingface.co/5CD-AI/Vintern-1B-v3_5
"""
# Ensure UTF-8 console output (fixes UnicodeEncodeError on Windows PowerShell)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# pip install ninja packaging wheel
# pip install flash-attn --no-build-isolation
# Khởi tạo timer
start_time = time.time()

# Chọn device (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runtime backend optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Using device:", device)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, use_thumbnail=False, pin_memory=False):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    # Fast path when using only one tile and no thumbnail
    if max_num == 1 and not use_thumbnail:
        pixel_values = transform(image).unsqueeze(0)
    else:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
    if pin_memory:
        pixel_values = pixel_values.pin_memory()
    return pixel_values

# Load model lên GPU
model_load_start = time.time()
model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=True,   # nếu đã cài flash-attn có thể đổi thành True
).to(device).eval()
model_load_end = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5", 
    trust_remote_code=True, 
    use_fast=False
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=r'C:\Users\ADMIN\Downloads\vintern_api\imgs\6.TKngknhnCMC_00001.png')
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--max_num', type=int, default=1)
    parser.add_argument('--use_thumbnail', action='store_true', default=False)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--do_sample', action='store_true', default=False)
    parser.add_argument('--repetition_penalty', type=float, default=2.5)
    parser.add_argument('--question', type=str, default='<image>\nTrích xuất thông tin chính trong ảnh và trả về dạng markdown.')
    parser.add_argument('--compile', action='store_true', default=False)
    args = parser.parse_args()

    pin_mem = device.type == 'cuda'

    # Validate input size for this model family (fallback to 448 if incompatible)
    valid_input_size = args.input_size
    try:
        # Many InternVL/Vintern checkpoints expect 448 per tile
        if args.input_size != 448:
            print(f"[warn] input_size {args.input_size} may be incompatible; falling back to 448 for stability.")
            valid_input_size = 448
    except Exception:
        valid_input_size = 448

    # Image preprocessing and non-blocking GPU transfer
    pixel_values = load_image(
        args.image,
        input_size=valid_input_size,
        max_num=args.max_num,
        use_thumbnail=args.use_thumbnail,
        pin_memory=pin_mem
    )
    pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
    pixel_values = pixel_values.to(device=device, dtype=torch.float16, non_blocking=True)

    # Optional compile for speedup (PyTorch 2.x). Fallback silently if unsupported.
    if args.compile:
        try:
            model_forward = model.forward
            model.forward = torch.compile(model_forward, mode='reduce-overhead', fullgraph=False)  # type: ignore
        except Exception:
            pass

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty
    )

    with torch.inference_mode():
        response, history = model.chat(
            tokenizer,
            pixel_values,
            args.question,
            generation_config,
            history=None,
            return_history=True
        )

    print(f'User: {args.question}\nAssistant: {response}')

    end_time = time.time()
    print(f'Model load: {model_load_end - model_load_start:.2f}s  |  Total: {end_time - start_time:.2f}s')

    del pixel_values
    if device.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()