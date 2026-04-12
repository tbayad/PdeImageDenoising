from dataclasses import dataclass, field
import numpy as np
from metrics import ssim, psnr
from typing import Dict, Optional


@dataclass
class ImageSample:
    image_id: int
    noise_type: str
    original: np.ndarray
    noisy: np.ndarray
    denoised: Optional[np.ndarray] = None
    metrics: Dict[str, Dict[str, Optional[float]]] = field(
        default_factory=lambda: {
            "before": {"ssim": None, "psnr": None},
            "after": {"ssim": None, "psnr": None},
        }
    )

    def compute_before_metrics(self) -> None:
        self.metrics["before"]["ssim"] = ssim(self.original, self.noisy)
        self.metrics["before"]["psnr"] = psnr(self.original, self.noisy)

    def update_denoised(self, denoised_image: np.ndarray) -> None:
        self.denoised = denoised_image
        self.metrics["after"]["ssim"] = ssim(self.original, denoised_image)
        self.metrics["after"]["psnr"] = psnr(self.original, denoised_image)
