
from typing import List
from datatype import ImageSample
import matplotlib.pyplot as plt
from io import BytesIO
import imageio

from utils.functions import download_dataset, load_images, build_samples
from utils.constants import ORIGINAL_IMAGE_PATH, SALT_PEPPER_NOISE_PATH
from diffusion.PeronaMalik import PeronaMalikDiffusion



def main() -> None:
    number_of_images = 10
    download_dataset()

    originals = load_images(ORIGINAL_IMAGE_PATH, number_of_images)
    if not originals:
        print("No original images loaded.")
        return

    samples: List[ImageSample] = []
    noise_type = "salt_and_pepper"
    noise_path = SALT_PEPPER_NOISE_PATH
    noisy_images = load_images(noise_path, number_of_images)
    samples.extend(build_samples(originals, noisy_images, noise_type))

    lambda_ = 20
    sigma = 1
    stepsize = 0.2
    n_steps = 5
    
    noisy_images = [noisy_images[6], noisy_images[3], noisy_images[2], noisy_images[8]]
    for i, noisy in enumerate(noisy_images):
        frames = []
        pm = PeronaMalikDiffusion(
            image=noisy,
            lambda_=lambda_,
            sigma=sigma,
            stepsize=stepsize,
            n_steps=n_steps,
        )
        pm.run(verbose=False)


        ig, ax = plt.subplots()
        plt.figure(figsize=(4,4))
        plt.imshow(pm.compute_edges(noisy), cmap="gray")
        plt.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()

        buf.seek(0)
        frames.append(imageio.v2.imread(buf))


        for image in pm.history:
            frame = pm.compute_edges(image)
    
            plt.figure(figsize=(4,4))
            plt.imshow(frame, cmap="gray")
            plt.axis("off")

            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close()

            buf.seek(0)
            frames.append(imageio.v2.imread(buf))

        imageio.mimsave(f"gifs/diffusion_{i}.gif", frames, fps=10, loop=0)


if __name__ == "__main__":
    main()