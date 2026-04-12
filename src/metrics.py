import numpy as np


def psnr(original: np.ndarray, noisy: np.ndarray) -> float:
    mse = ((original - noisy) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def ssim(original: np.ndarray, noisy: np.ndarray) -> float:
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    original = original.astype(np.float64)
    noisy = noisy.astype(np.float64)

    mu_x = original.mean()
    mu_y = noisy.mean()
    sigma_x = original.var()
    sigma_y = noisy.var()
    sigma_xy = ((original - mu_x) * (noisy - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0