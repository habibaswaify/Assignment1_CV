import numpy as np
from convolution import conv2d_sep, gaussian_kernel

#finite difference
Dx = np.array([[1.0, 0.0, -1.0]])
Dy = Dx.T #transpose 

def gradient_magnitude(gx, gy):
    return np.sqrt(gx**2 + gy**2)

def to_gray(img):
    if img.ndim == 2:
        return img
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])   #standard formula for gray

def finite_diff(gray, pad_mode="constant"):
    return conv2d_sep(gray, Dx, pad_mode), conv2d_sep(gray, Dy, pad_mode)


def dog_filters(size, sigma):
    G = gaussian_kernel(size, sigma)
    return conv2d_sep(G, Dx), conv2d_sep(G, Dy)


def dog_diff(gray, dogx, dogy, pad_mode="constant"):
    return conv2d_sep(gray, dogx, pad_mode), conv2d_sep(gray, dogy, pad_mode)


def signed_to_display(x):
    m = np.abs(x).max()
    return np.full_like(x, 0.5) if m == 0 else x / (2 * m) + 0.5