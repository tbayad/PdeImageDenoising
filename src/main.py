import os
import kaggle
import imageio.v2 as iio
import numpy as np

from pmdiffusion import PeronaMalikDiffusion


DATASET_PATH = "data/"

GAUSSIAN_NOISE_PATH = DATASET_PATH + "dataset/noises/gaussian/"
SALT_PEPPER_NOISE_PATH = DATASET_PATH + "dataset/noises/salt_and_pepper/"  # Fixed path
SPECKLE_NOISE_PATH = DATASET_PATH + "dataset/noises/speckle/"
ORIGINAL_IMAGE_PATH = DATASET_PATH + "dataset/original/"



GAUSSIAN_NOISE = "gaussian"
SALT_PEPPER_NOISE =  "salt_pepper"
SPECKLE_NOISE =  "speckle"
ORIGINAL_IMAGE = "original"







def rgb2gray(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.5870 + rgb[:, :, 2] * 0.1140


def psnr(original, noisy):
    mse = ((original - noisy) ** 2).mean()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr_value = 10 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def ssim(original, noisy):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    original = original.astype(np.float64)
    noisy = noisy.astype(np.float64)

    mu_x = original.mean()
    mu_y = noisy.mean()
    sigma_x = original.var()
    sigma_y = noisy.var()
    sigma_xy = ((original - mu_x) * (noisy - mu_y)).mean()

    ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_value


def download_dataset(n_imgs: int = 10):
     if not os.path.exists(DATASET_PATH):
       kaggle.api.dataset_download_files("goutham1208/multi-noises-for-image-denoising", path=DATASET_PATH, unzip=True)


class Image:
    def __init__(self, data: np.ndarray | None = None, psnr: float | None = None, ssim: float | None = None):
        self.data = data
        self.psnr = psnr
        self.ssim = ssim
    

class ImagesSet:
    def __init__(self):
        self.gaussian = []
        self.salt_pepper = []
        self.speckle = []
        self.original = []

    def append_image(self, img: np.ndarray,  img_type: str):
        if img_type == GAUSSIAN_NOISE:
             self.gaussian.append(Image(data = img))
        elif img_type == SALT_PEPPER_NOISE:
             self.salt_pepper.append(Image(data = img))
        elif img_type == SPECKLE_NOISE:
             self.speckle.append(Image(data = img))
        elif img_type == ORIGINAL_IMAGE:
             self.original.append(Image(data = img))



datasets_map = {
        GAUSSIAN_NOISE_PATH: GAUSSIAN_NOISE,
        SALT_PEPPER_NOISE_PATH: SALT_PEPPER_NOISE,
        SPECKLE_NOISE_PATH: SPECKLE_NOISE,
        ORIGINAL_IMAGE_PATH: ORIGINAL_IMAGE
    }


def load_images(image_set: ImagesSet, number_of_images: int):
    for key in datasets_map:
        count = 0 
        for filename in os.listdir(key):
            if count == number_of_images -1:
                    break
            count += 1
            image = iio.imread(os.path.join(key, filename))
            image = image.astype('float32') / 255.0
            if image.shape[-1] == 3:
                image = rgb2gray(image)
            image_set.append_image(img=image, img_type=datasets_map[key])
    

def compute_metrics(image_set: ImagesSet):
    for noise_type in ["gaussian", "salt_pepper", "speckle"]:
        noisy_imgs = getattr(image_set, noise_type)
        for i, (original, noisy) in enumerate(zip(image_set.original, noisy_imgs)):
            ssim_value = ssim(original.data, noisy.data)
            psnr_value = psnr(original.data, noisy.data)

            noisy_imgs[i].psnr = psnr_value
            noisy_imgs[i].ssim = ssim_value 
        


def load_original(image_set: ImagesSet, number_of_images: int = 10):
    count = 0
    for filename in os.listdir(ORIGINAL_IMAGE_PATH):
        if count == number_of_images -1:
                break
        count += 1
        image = iio.imread(os.path.join(ORIGINAL_IMAGE_PATH, filename))
        image = image.astype('float32') / 255.0
        if image.shape[-1] == 3:
            image = rgb2gray(image)
        image_set.append_image(img=image, img_type=ORIGINAL_IMAGE)



def peform_pmdiff(image_set: ImagesSet, denoised_set: ImagesSet):
    for noise_type in ["gaussian", "salt_pepper", "speckle"]:
        noisy_imgs = getattr(image_set, noise_type)
        for i, (original, noisy) in enumerate(zip(image_set.original, noisy_imgs)):
            pm = PeronaMalikDiffusion(
                    image=noisy.data,
                    lambda_=20,
                    sigma=1,
                    stepsize=0.2,
                    n_steps=20
                )     
            pm.run(verbose=False, plot_every=10000)
            pm.plot_evolution()
            denoised_img = pm.history[-1]
            denoised_set.append_image(img=denoised_img, img_type=noise_type)

            ssim_value = ssim(original.data, denoised_img)
            psnr_value = psnr(original.data, denoised_img)
            
            denoised_set[i].psnr = psnr_value
            denoised_set[i].ssim = ssim_value
    

def main():
    
    number_of_images  = 10
    img_set = ImagesSet()
    denoised_set = ImagesSet()

    download_dataset(number_of_images)
    load_images(img_set, number_of_images)
    compute_metrics(img_set)
    load_original(denoised_set,number_of_images)
    peform_pmdiff(img_set, denoised_set)
    
    


if __name__ == "__main__":
    main()
