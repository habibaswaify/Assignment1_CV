import os
import numpy as np
from convolution import conv2d_sep, gaussian_kernel

def get_gaussian_kernel(sigma):
    """Generates a normalized 2D Gaussian kernel with odd window dimension matching sigma."""
    radius = int(np.ceil(3.0 * sigma))
    size = 2 * radius + 1
    ax = np.arange(size) - radius
    g = np.exp(-ax**2 / (2.0 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)

def filter_image(img, kernel, pad_mode="reflect"):
    """
    Applies separable convolution using reflection padding.
    Handles both single-channel (grayscale) and 3-channel (color) images.
    """
    if img.ndim == 2:
        return conv2d_sep(img, kernel, pad_mode=pad_mode)
    elif img.ndim == 3:
        channels = [conv2d_sep(img[:, :, c], kernel, pad_mode=pad_mode) for c in range(img.shape[2])]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError("Image must have 2 or 3 dimensions.")

def low_pass(img, sigma):
    kernel = get_gaussian_kernel(sigma)
    return filter_image(img, kernel, pad_mode="reflect")

def high_pass(img, sigma):
    return img - low_pass(img, sigma)

def make_hybrid(img1, img2, sigma_low, sigma_high):

    lp = low_pass(img1, sigma_low)
    hp = high_pass(img2, sigma_high)
    hybrid = lp + hp
    return lp, hp, np.clip(hybrid, 0.0, 1.0)


def compute_fft_log_magnitude(img):
    """Computes the log magnitude of the 2D FFT on a grayscale image, exactly as required."""
    # Convert to grayscale if the image has 3 channels
    if img.ndim == 3:
        gray_image = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        gray_image = img.copy()
        
    # Assignment exact formula
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image))))
