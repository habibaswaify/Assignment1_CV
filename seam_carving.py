import numpy as np
from filters_helpers import gradient_magnitude,to_gray

def compute_energy_from_gradient(img) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    gray = to_gray(img) if img.ndim == 3 else img

    if gray.ndim == 3:                
        gray = gray.mean(axis=2)

    gy, gx = np.gradient(gray)
    return gradient_magnitude(gx, gy)

def find_seam(energy) -> np.ndarray:
    '''using tabulation bottom-up approach'''
    h, w = energy.shape
    seam_energies = energy.copy()    
    # back pointer -> the tabulation variable
    backptr = np.zeros((h, w), dtype=int)

    for row in range(1, h):               
        for col in range(w):        
            col_left = max(col - 1, 0)      
            col_right = min(col + 1, w - 1)  
            col_range = range(col_left, col_right + 1) 

            # minimum cost in the parent column
            parent_col = min(col_range, key=lambda col_i: seam_energies[row - 1, col_i])   
            backptr[row, col] = parent_col   
            # compute accumulative energy current + parent    
            seam_energies[row, col] += seam_energies[row - 1, parent_col]   
            
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

    
def reduce_img_size(img, target_width: int,highlight_color:tuple = None) -> tuple:
    h,w = img.shape[:2]
    reduced_img = np.array(img, dtype=np.float64) 
    highlighted_img = np.array(img, dtype=np.float64) 
    iter_num = 0
    
    if target_width > img.shape[1]:
        raise ValueError("target_width must not exceed the current width")
    if highlight_color ==None:
        highlight_color = (255,0,0)
        
    # original column of every remaining pixel
    col_idx = np.tile(np.arange(w), (h, 1))
    rows = np.arange(h)
    
    while reduced_img.shape[1] > target_width:
        # print(f"Iteration: {iter_num}")
        
        energy = compute_energy_from_gradient(reduced_img)
        seam = find_seam(energy)
        # shrink width by 1 each iteration
        reduced_img = remove_seam(reduced_img, seam)    
        
        # map seam to original coordinates and shrink the index map
        original_cols = col_idx[rows,seam]
        highlight_seam(highlighted_img,original_cols,highlight_color)
        col_idx = remove_seam(col_idx,seam) 
    
        iter_num+=1
    
    return reduced_img,highlighted_img