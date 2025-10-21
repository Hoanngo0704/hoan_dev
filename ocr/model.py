import torch
from transformers import AutoModel, AutoTokenizer

class OCRModel:
    def __init__(self, model_id: str, allow_flash_attn: bool = True):
        self.model_id = model_id
        self.allow_flash_attn = allow_flash_attn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = None
        self.tok = None

    @property
    def on_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def device_str(self) -> str:
        return str(self.device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self):
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=self.allow_flash_attn,
        ).to(self.device).eval()

        self.tok = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )

    def prepare_tensor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # tối ưu memory format + dtype + non_blocking
        return pixel_values.contiguous(memory_format=torch.channels_last)\
                           .to(device=self.device, dtype=torch.float16, non_blocking=True)

    def chat(self, pixel_values, question: str, **gen_cfg):
        with torch.inference_mode():
            response, _ = self.model.chat(
                self.tok,
                pixel_values,
                question,
                gen_cfg,
                history=None,
                return_history=True
            )
        return response
