from collections.abc import Callable

import numpy as np
from filters_helpers import (
    dog_diff,
    dog_filters,
    finite_diff,
    gradient_magnitude,
    to_gray,
)


def compute_energy_from_gradient(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(to_gray(gray), dtype=np.float64)
    gy, gx = np.gradient(gray)
    return gradient_magnitude(gx, gy)


def compute_energy_from_finite_diff(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(to_gray(gray), dtype=np.float64)
    gx, gy = finite_diff(gray)
    return gradient_magnitude(gx, gy)


def compute_energy_from_dog(
    gray: np.ndarray, size: int = 5, sigma: float = 1.0
) -> np.ndarray:
    gray = np.asarray(to_gray(gray), dtype=np.float64)
    dogx, dogy = dog_filters(size, sigma)
    gx, gy = dog_diff(gray, dogx, dogy)
    return gradient_magnitude(gx, gy)


def find_seam(energy) -> np.ndarray:
    '''using tabulation bottom-up approach'''
    h, w = energy.shape
    seam_energies = energy.copy()    
    # back pointer -> the tabulation variable
    backptr = np.zeros((h, w), dtype=int)

    for row in range(1, h):
        prev = seam_energies[row - 1]
        cols = np.arange(w)
        col_left = np.maximum(cols - 1, 0)
        col_right = np.minimum(cols + 1, w - 1)
        candidates = np.stack((prev[col_left], prev[cols], prev[col_right]), axis=1)

        # minimum cost in the parent column
        parent_choice = np.argmin(candidates, axis=1)
        backptr[row] = np.where(
            parent_choice == 0,
            col_left,
            np.where(parent_choice == 1, cols, col_right),
        )
        # compute accumulative energy current + parent
        seam_energies[row] += np.choose(parent_choice, (candidates[:, 0], candidates[:, 1], candidates[:, 2]))
            
    seam = np.zeros(h, dtype=int)
    # get the lower cost seam
    seam[-1] = np.argmin(seam_energies[-1])
    for row in range(h - 1, 0, -1):
        # backtracking step: get the cheapest cost for the parent
        seam[row - 1] = backptr[row, seam[row]]
    return seam        

def remove_seam(img, seam: np.ndarray):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    # remove the values indexed by the seam positions
    mask[np.arange(h), seam] = False
    if img.ndim == 3:
        return img[mask].reshape(h, w - 1, img.shape[2])
    return img[mask].reshape(h, w - 1)

def highlight_seam(img, seam: np.ndarray, highlight_color:tuple):
    img[np.arange(img.shape[0]), seam] = highlight_color
    return img

    
def carve_seams(
    img,
    target_width_ratio: float,
    highlight_color: tuple = None,
    energy_function: Callable[[np.ndarray], np.ndarray] = compute_energy_from_gradient,
    vertical: bool = True,
    horizontal: bool = False,
) -> tuple:
    h, w = img.shape[:2]
    reduced_img = np.array(img, dtype=np.float64)
    highlighted_img = np.array(img, dtype=np.float64)

    if target_width_ratio < 0:
        raise ValueError("target_width_ratio must be positive")
    if target_width_ratio > 1:
        raise ValueError("target_width_ratio must not exceed the current width")
    if highlight_color is None:
        highlight_color = (255, 0, 0)

    # track each remaining pixel's original location for seam highlighting
    original_indices = np.arange(h * w).reshape(h, w)

    for is_horizontal in (False, True):
        if not (horizontal if is_horizontal else vertical):
            continue

        working_img = np.swapaxes(reduced_img, 0, 1) if is_horizontal else reduced_img
        working_indices = (
            np.swapaxes(original_indices, 0, 1)
            if is_horizontal
            else original_indices
        )
        target_width = np.floor((1 - target_width_ratio) * working_img.shape[1])
        rows = np.arange(working_img.shape[0])

        while working_img.shape[1] > target_width:
            energy = energy_function(to_gray(working_img))
            seam = find_seam(energy)
            removed_indices = working_indices[rows, seam]
            original_rows, original_cols = np.divmod(removed_indices, w)
            highlighted_img[original_rows, original_cols] = highlight_color
            working_img = remove_seam(working_img, seam)
            working_indices = remove_seam(working_indices, seam)

        if is_horizontal:
            reduced_img = np.swapaxes(working_img, 0, 1)
            original_indices = np.swapaxes(working_indices, 0, 1)
        else:
            reduced_img = working_img
            original_indices = working_indices

    return reduced_img, highlighted_img