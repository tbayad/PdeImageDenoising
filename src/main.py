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

