import numpy as np
import os
import kaggle
import imageio.v2 as iio
from typing import List, Dict

from datatype import ImageSample
from diffusion.PeronaMalik import PeronaMalikDiffusion
from utils.constants import DATASET_PATH


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.5870 + rgb[:, :, 2] * 0.1140


def download_dataset() -> None:
    if not os.path.exists(DATASET_PATH):
        kaggle.api.dataset_download_files(
            "goutham1208/multi-noises-for-image-denoising",
            path=DATASET_PATH,
            unzip=True,
        )



def load_images(path: str, max_images: int) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    if not os.path.isdir(path):
        print(f"Warning: folder not found: {path}")
        return images

    for filename in sorted(os.listdir(path))[:max_images]:
        image_path = os.path.join(path, filename)
        image = iio.imread(image_path).astype("float32") / 255.0
        if image.ndim == 3 and image.shape[-1] == 3:
            image = rgb2gray(image)
        images.append(image)

    return images



def build_samples(originals: List[np.ndarray], noisy_images: List[np.ndarray], noise_type: str) -> List[ImageSample]:
    samples: List[ImageSample] = []
    for image_id, (original, noisy) in enumerate(zip(originals, noisy_images)):
        sample = ImageSample(
            image_id=image_id,
            noise_type=noise_type,
            original=original,
            noisy=noisy,
        )
        sample.compute_before_metrics()
        samples.append(sample)
    return samples


def apply_diffusion(
    samples: List[ImageSample],
    lambda_: float = 20.0,
    sigma: float = 1.0,
    stepsize: float = 0.2,
    n_steps: int = 5,
) -> None:
    for sample in samples:
        pm = PeronaMalikDiffusion(
            image=sample.noisy,
            lambda_=lambda_,
            sigma=sigma,
            stepsize=stepsize,
            n_steps=n_steps,
        )
        pm.run(verbose=False)
        pm.plot_evolution(sample.image_id, sample.noise_type)
        denoised_image = pm.history[-1]
        sample.update_denoised(denoised_image)


def summarize(samples: List[ImageSample]) -> None:
    grouped: Dict[str, List[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.noise_type, []).append(sample)

    for noise_type, group in grouped.items():
        before_ssim = [s.metrics["before"]["ssim"] for s in group if s.metrics["before"]["ssim"] is not None]
        before_psnr = [s.metrics["before"]["psnr"] for s in group if s.metrics["before"]["psnr"] is not None]
        after_ssim = [s.metrics["after"]["ssim"] for s in group if s.metrics["after"]["ssim"] is not None]
        after_psnr = [s.metrics["after"]["psnr"] for s in group if s.metrics["after"]["psnr"] is not None]

        print(f"Noise type: {noise_type}")
        if before_ssim:
            print(f"  before: avg SSIM={np.mean(before_ssim):.4f}, avg PSNR={np.mean(before_psnr):.4f}")
        if after_ssim:
            print(f"  after : avg SSIM={np.mean(after_ssim):.4f}, avg PSNR={np.mean(after_psnr):.4f}")