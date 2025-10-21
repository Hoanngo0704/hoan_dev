from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 448

def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
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

def to_tensor_one_tile(pil_img: Image.Image, input_size=DEFAULT_INPUT_SIZE, pin_memory=False) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    t = transform(pil_img).unsqueeze(0)
    if pin_memory: t = t.pin_memory()
    return t
