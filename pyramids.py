import numpy as np
from convolution import conv2d_sep

BINOMIAL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
BINOMIAL_2D = np.outer(BINOMIAL_1D, BINOMIAL_1D)
BINOMIAL_COL = BINOMIAL_1D.reshape(-1, 1)
BINOMIAL_ROW = BINOMIAL_1D.reshape(1, -1)


def blur_binomial(img, pad_mode="reflect"):
    if img.ndim == 2:
        return conv2d_sep(img.astype(np.float64), BINOMIAL_2D, pad_mode=pad_mode)
    elif img.ndim == 3:
        channels = [
            conv2d_sep(img[:, :, c].astype(np.float64), BINOMIAL_2D, pad_mode=pad_mode)
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

def multiresolution_blend(img_a, img_b, mask, levels=5):

    ga_pyr = build_gaussian_pyramid(img_a, levels=levels)
    la_pyr = build_laplacian_pyramid(ga_pyr)
    
    gb_pyr = build_gaussian_pyramid(img_b, levels=levels)
    lb_pyr = build_laplacian_pyramid(gb_pyr)

    gm_pyr = build_gaussian_pyramid(mask.astype(np.float64), levels=levels)

    l_blend = []
    for k in range(len(la_pyr)):
        m_k = gm_pyr[k]
        if img_a.ndim == 3 and m_k.ndim == 2:
            m_k = m_k[:, :, None]

        blended_k = m_k * la_pyr[k] + (1.0 - m_k) * lb_pyr[k]
        l_blend.append(blended_k)

    reconstructed = reconstruct_laplacian_pyramid(l_blend)
    return np.clip(reconstructed, 0.0, 1.0)