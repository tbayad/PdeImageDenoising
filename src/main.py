# import os
# import kaggle
# import imageio.v2 as iio
# import numpy as np
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional

# from pmdiffusion import PeronaMalikDiffusion

# DATASET_PATH = "data/"
# GAUSSIAN_NOISE_PATH = DATASET_PATH + "dataset/noises/gaussian/"
# SALT_PEPPER_NOISE_PATH = DATASET_PATH + "dataset/noises/salt_and_pepper/"
# SPECKLE_NOISE_PATH = DATASET_PATH + "dataset/noises/speckle/"
# ORIGINAL_IMAGE_PATH = DATASET_PATH + "dataset/original/"

# NOISE_PATHS = {
#     "gaussian": GAUSSIAN_NOISE_PATH,
#     "salt_and_pepper": SALT_PEPPER_NOISE_PATH,
#     "speckle": SPECKLE_NOISE_PATH,
# }


# @dataclass
# class ImageSample:
#     image_id: int
#     noise_type: str
#     original: np.ndarray
#     noisy: np.ndarray
#     denoised: Optional[np.ndarray] = None
#     metrics: Dict[str, Dict[str, Optional[float]]] = field(
#         default_factory=lambda: {
#             "before": {"ssim": None, "psnr": None},
#             "after": {"ssim": None, "psnr": None},
#         }
#     )

#     def compute_before_metrics(self) -> None:
#         self.metrics["before"]["ssim"] = ssim(self.original, self.noisy)
#         self.metrics["before"]["psnr"] = psnr(self.original, self.noisy)

#     def update_denoised(self, denoised_image: np.ndarray) -> None:
#         self.denoised = denoised_image
#         self.metrics["after"]["ssim"] = ssim(self.original, denoised_image)
#         self.metrics["after"]["psnr"] = psnr(self.original, denoised_image)


# def rgb2gray(rgb: np.ndarray) -> np.ndarray:
#     return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.5870 + rgb[:, :, 2] * 0.1140


# def psnr(original: np.ndarray, noisy: np.ndarray) -> float:
#     mse = ((original - noisy) ** 2).mean()
#     if mse == 0:
#         return float("inf")
#     return 10 * np.log10(1.0 / mse)


# def ssim(original: np.ndarray, noisy: np.ndarray) -> float:
#     C1 = (0.01 * 1.0) ** 2
#     C2 = (0.03 * 1.0) ** 2

#     original = original.astype(np.float64)
#     noisy = noisy.astype(np.float64)

#     mu_x = original.mean()
#     mu_y = noisy.mean()
#     sigma_x = original.var()
#     sigma_y = noisy.var()
#     sigma_xy = ((original - mu_x) * (noisy - mu_y)).mean()

#     numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
#     denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
#     return float(numerator / denominator) if denominator != 0 else 0.0


# def download_dataset() -> None:
#     if not os.path.exists(DATASET_PATH):
#         kaggle.api.dataset_download_files(
#             "goutham1208/multi-noises-for-image-denoising",
#             path=DATASET_PATH,
#             unzip=True,
#         )



# def load_images(path: str, max_images: int) -> List[np.ndarray]:
#     images: List[np.ndarray] = []
#     if not os.path.isdir(path):
#         print(f"Warning: folder not found: {path}")
#         return images

#     for filename in sorted(os.listdir(path))[:max_images]:
#         image_path = os.path.join(path, filename)
#         image = iio.imread(image_path).astype("float32") / 255.0
#         if image.ndim == 3 and image.shape[-1] == 3:
#             image = rgb2gray(image)
#         images.append(image)

#     return images



# def build_samples(originals: List[np.ndarray], noisy_images: List[np.ndarray], noise_type: str) -> List[ImageSample]:
#     samples: List[ImageSample] = []
#     for image_id, (original, noisy) in enumerate(zip(originals, noisy_images)):
#         sample = ImageSample(
#             image_id=image_id,
#             noise_type=noise_type,
#             original=original,
#             noisy=noisy,
#         )
#         sample.compute_before_metrics()
#         samples.append(sample)
#     return samples


# def apply_diffusion(
#     samples: List[ImageSample],
#     lambda_: float = 20.0,
#     sigma: float = 1.0,
#     stepsize: float = 0.2,
#     n_steps: int = 5,
# ) -> None:
#     for sample in samples:
#         pm = PeronaMalikDiffusion(
#             image=sample.noisy,
#             lambda_=lambda_,
#             sigma=sigma,
#             stepsize=stepsize,
#             n_steps=n_steps,
#         )
#         pm.run(verbose=False)
#         denoised_image = pm.history[-1]
#         sample.update_denoised(denoised_image)


# def summarize(samples: List[ImageSample]) -> None:
#     grouped: Dict[str, List[ImageSample]] = {}
#     for sample in samples:
#         grouped.setdefault(sample.noise_type, []).append(sample)

#     for noise_type, group in grouped.items():
#         before_ssim = [s.metrics["before"]["ssim"] for s in group if s.metrics["before"]["ssim"] is not None]
#         before_psnr = [s.metrics["before"]["psnr"] for s in group if s.metrics["before"]["psnr"] is not None]
#         after_ssim = [s.metrics["after"]["ssim"] for s in group if s.metrics["after"]["ssim"] is not None]
#         after_psnr = [s.metrics["after"]["psnr"] for s in group if s.metrics["after"]["psnr"] is not None]

#         print(f"Noise type: {noise_type}")
#         if before_ssim:
#             print(f"  before: avg SSIM={np.mean(before_ssim):.4f}, avg PSNR={np.mean(before_psnr):.4f}")
#         if after_ssim:
#             print(f"  after : avg SSIM={np.mean(after_ssim):.4f}, avg PSNR={np.mean(after_psnr):.4f}")


from typing import List
from datatype import ImageSample

from utils.functions import download_dataset, load_images, build_samples, apply_diffusion, summarize
from utils.constants import ORIGINAL_IMAGE_PATH, NOISE_PATHS


def main() -> None:
    number_of_images = 10
    download_dataset()

    originals = load_images(ORIGINAL_IMAGE_PATH, number_of_images)
    if not originals:
        print("No original images loaded.")
        return

    samples: List[ImageSample] = []
    for noise_type, path in NOISE_PATHS.items():
        noisy_images = load_images(path, number_of_images)
        if not noisy_images:
            print(f"No noisy images loaded for {noise_type}.")
            continue
        samples.extend(build_samples(originals, noisy_images, noise_type))

    if not samples:
        print("No samples available for diffusion.")
        return

    apply_diffusion(samples)
    summarize(samples)


if __name__ == "__main__":
    main()

