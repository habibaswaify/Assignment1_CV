import numpy as np


def padding(img, ph, pw, pad_mode):    #part 1 needs 0, 2 needs reflect modr
    return np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode)


def conv2d_loop(img, k, pad_mode="constant"):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2   #padding
    padded = padding(img.astype(np.float64), ph, pw, pad_mode)
    kf = k[::-1, ::-1]  #convolution= correlation flipped -- flip x flip y
    H, W = img.shape
    out = np.zeros((H, W))
    for i in range(H):  #looping on h dim 
        for j in range(W):  #looping on w dim
            s = 0.0
            for u in range(kh):
                for v in range(kw):
                    s += padded[i + u, j + v] * kf[u, v]    #weighted sum
            out[i, j] = s
    return out


def conv2d_vec(img, k, pad_mode="constant"):
    #sliding kernel
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    padded = padding(img.astype(np.float64), ph, pw, pad_mode)
    kf = k[::-1, ::-1]
    H, W = img.shape
    out = np.zeros((H, W))
    for u in range(kh):
        for v in range(kw):
            out += kf[u, v] * padded[u:u + H, v:v + W]
    return out


def separate(k, tol=1e-10): #svd
    U, S, Vt = np.linalg.svd(k)
    if S[0] == 0:
        return None
    if len(S) > 1 and S[1] > tol * S[0]:    #for small values (rounding error)- to check its sep
        return None
    s = np.sqrt(S[0])
    return (U[:, 0] * s).reshape(-1, 1), (Vt[0] * s).reshape(1, -1)


def conv2d_sep(img, k, pad_mode="constant"):
    sep= separate(k)
    if sep is None:
        return conv2d_vec(img, k, pad_mode) #otherwise use vectorized
    col, row= sep
    return conv2d_vec(conv2d_vec(img, col, pad_mode), row, pad_mode)    #convolution col & row


def gaussian_kernel(size, sigma):   #2d gaussian
    ax = np.arange(size) - size // 2    #create 1d, find center, subtract to normalise
    g = np.exp(-ax**2 / (2 * sigma**2)) #gaussian fn
    g /= g.sum()    #norm
    return np.outer(g, g)   #product

##check
if __name__ == "__main__":
    from scipy.signal import convolve2d
    rng = np.random.default_rng(0)
    for _ in range(20):
        img = rng.random((rng.integers(5, 20), rng.integers(5, 20)))    #random images of fixed params for tesr
        kh, kw = rng.choice([1, 3, 5, 7]), rng.choice([1, 3, 5, 7])
        k_rand = rng.random((kh, kw))
        k_sep = np.outer(rng.random(kh), rng.random(kw))
        for k in (k_rand, k_sep):
            ref = convolve2d(img, k, mode="same")
            for f in (conv2d_loop, conv2d_vec, conv2d_sep):
                assert np.allclose(f(img, k), ref), (f.__name__, k.shape)
    print("all three versions match scipy")