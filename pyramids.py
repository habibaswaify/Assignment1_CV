import numpy as np
from convolution import conv2d_sep

BINOMIAL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
BINOMIAL_COL = BINOMIAL_1D.reshape(-1, 1)
BINOMIAL_ROW = BINOMIAL_1D.reshape(1, -1)


def blur_binomial(img, pad_mode="reflect"):
    if img.ndim == 2:
        return conv2d_sep(img, BINOMIAL_COL, pad_mode=pad_mode)
    elif img.ndim == 3:
        channels = [
            conv2d_sep(img[:, :, c], BINOMIAL_COL, pad_mode=pad_mode)
            for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")


def reduce(img):
    blurred = blur_binomial(img, pad_mode="reflect")
    return blurred[::2, ::2]


def expand(img, target_shape=None):
    h, w = img.shape[:2]
    out_h, out_w = h * 2, w * 2

    if img.ndim == 2:
        upsampled = np.zeros((out_h, out_w), dtype=np.float64)
        upsampled[::2, ::2] = img
    else:
        channels = img.shape[2]
        upsampled = np.zeros((out_h, out_w, channels), dtype=np.float64)
        upsampled[::2, ::2, :] = img

    expanded = 4.0 * blur_binomial(upsampled, pad_mode="reflect")

    if target_shape is not None:
        expanded = expanded[: target_shape[0], : target_shape[1]]

    return expanded


def build_gaussian_pyramid(img, levels=4):
    G = [img.astype(np.float64)]
    for _ in range(levels):
        G.append(reduce(G[-1]))
    return G


def build_laplacian_pyramid(gaussian_pyramid):

    L = []
    for k in range(len(gaussian_pyramid) - 1):
        target_shape = gaussian_pyramid[k].shape[:2]
        expanded_next = expand(gaussian_pyramid[k + 1], target_shape=target_shape)
        L.append(gaussian_pyramid[k] - expanded_next)
    L.append(gaussian_pyramid[-1])
    return L


def reconstruct_laplacian_pyramid(laplacian_pyramid):

    curr = laplacian_pyramid[-1]
    for k in range(len(laplacian_pyramid) - 2, -1, -1):
        target_shape = laplacian_pyramid[k].shape[:2]
        curr = laplacian_pyramid[k] + expand(curr, target_shape=target_shape)
    return curr