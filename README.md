# Assignment1_CV


# Part 1: Filters and Edges

Code is in `convolution.py`, `filters_helpers.py` and `part1.ipynb`. 
From 1.2 on, every blur and derivative runs on my separable convolution (`conv2d_sep`).
Images are grayscale and max 1000 px on the longer side.

## 1.1 Convolution from scratch

2D convolution were written in 3 versions. 
All of them flip the filter (convolution is correlation with the flipped filter) and zero pad, so the output is the same size as the input.

- **Loop:** loops over every pixel and every filter entry.
- **Vectorized:** no loop over pixels. It loops over the filter entries only, adding filter value × a shifted copy of the whole padded image each time.
- **Separable:** checks if the filter is rank 1 using its SVD (second singular value ~ 0 compared to the first, rounding tolerance 1e-10). If yes, it convolves with the column then the row. If not, it uses the vectorized version.


**Check:**   
50 trials were ran on small random images (sides: 5~24). Each trial used a random non-separable filter and a random separable one (outer product of a random column and row), with random odd sizes from 1 to 9. All three versions matched `scipy.signal.convolve2d(mode="same")` (`np.allclose`, atol 1e-10), so they agree to rounding error "tol".

**Timing:**   
Random 256 × 256 image, Gaussian filters from 3 × 3 to 31 × 31 (sigma = size / 6). All of them are rank 1, so the separable version always takes its fast path.

```  
  size   loop (s)   vectorized (s)   separable (s)
  3x3       0.187           0.0004          0.0005
  5x5       0.462           0.0011          0.0006
  7x7       0.888           0.0020          0.0007
  9x9       1.478           0.0047          0.0015
 11x11      2.108           0.0051          0.0010
 13x13      2.918           0.0071          0.0012
 15x15      3.785           0.0089          0.0013
 17x17      4.818           0.0116          0.0014
 19x19      8.403           0.0161          0.0019
 21x21     10.031           0.0199          0.0029
 23x23     13.211           0.0337          0.0019
 25x25     17.589           0.0319          0.0033
 27x27     18.808           0.0506          0.0044
 29x29     23.704           0.0409          0.0023
 31x31     26.248           0.0684          0.0030

```

![alt text](figs/timing.png)

**Why the times are different:**
- **Loop:** about H·W·k² multiply-adds, done one by one by the Python, so it is by far the slowest and grows with k², where time is 26.248 s.
- **Vectorized:** only k² Python iterations, and each one is a whole-image NumPy operation running in compiled code. The arithmetic is still O(H·W·k²), but each operation is much cheaper, so it is much faster than the loop, where time is 0.0684 s.
- **Separable:** two 1D passes cost 2k whole-image operations instead of k², so it grows linearly with k and the gap to the vectorized version gets bigger with filter size, where time is 0.0030 s. The SVD test costs almost nothing because the filter is small.

## 1.2 Finite difference operator

An edge is a sharp change in intensity, and derivatives measure change (in our case change from the px before and after). The grayscale image was filtered with Dx = [1 0 -1] and Dy = Dx transposed, the gradient magnitude was calculated and thresholded. Repeated for both example images (cameraman & facade). Derivative maps are shown with zero as mid-gray.

![alt text](figs/1_2_cameraman.png)

![\[FIGURE: figs/1_2_facade.png\]](figs/1_2_facade.png)

**Threshold:** swept T = 0.05, 0.1, 0.2, 0.3, 0.4 and picked T = 0.2 for both images.
- **Facade:** at 0.05 and 0.1 the wall and the shutters fill with speckle that hides the structure. At 0.3 and 0.4 the window outlines and the door arch start breaking up. At 0.2 the windows, balcony, arch and signs stay clean and the flat wall is black.
- **Cameraman:** Simillar results for camera man, at 0.05 and 0.1 grass and cameraman's coat are filled with speckles, at 0.3 and 0.4 distant edges start disappearing and cameraman's edges start breaking up and fine details aroud edges are gone. At 0.2 cameraman's edges are most outlides with little loss, downto the coat buttons and distant structures.    

**Lines along the frame:** with zero padding the convolution assumes black (0) outside the image. On the first and last row and column the filter compares a real pixel with a fake black one, so it sees a big step wherever the border is brighter than the threshold. Dx makes the lines on the left and right sides and Dy on the top and bottom. They are not real edges, only the padding - yet they correspond to an edge, and therefore theres a white aura around the image (a gray border is added for visibility). With reflect padding there is no fake step and the lines are gone.

![\[FIGURE: figs/1_2_frame_cameraman.png\]](figs/1_2_frame_cameraman.png)

![alt text](figs/1_2_corner_cameraman.png)


## 1.3 Derivative of Gaussian (DoG)

Derivatives amplify noise, so one solution is to blur before differentiating. Convolution is associative, so the blur and the derivative can be one filter. To get the two DoG filters, a 2D Gaussian (sigma = 2, 13 × 13) was built and the kernel was filtered with Dx and Dy. Each one was then applied to the image with a single convolution. Then the same steps as 1.2 were repeated.

![alt text](figs/1_3_filters.png)

![\[FIGURE: figs/1_3_cameraman.png\]](figs/1_3_cameraman.png)

![alt text](figs/1_3_cameraman.png)

**Parameters:** sigma = 2, T = 0.08 for the cameraman and T = 0.06 for the facade, again from a sweep (0.02, 0.04, 0.06, 0.08, 0.12). On the facade, 0.04 leaves blobs on the walls, 0.06 gives clean walls with continuous contours, and from 0.08 the door arch and thin contours start to disappear. T = 0.8, on cameraman image, achieves best tradeoff with clearest edges.

**Check:** one DoG convolution matches blur first then differentiate, up to about 0.002. That small difference is just from cutting the Gaussian to 13 × 13.



**How the maps differ from 1.2:**
- Smoother. The speckle on textured areas (grass, wall plaster, shutters) is gone.
- Edges are continuous and thicker, while 1.2 gives thin broken lines.
- Fine detail is lost. Only the main structure survives.
- Gradient values are smaller (due to smoothing), so the threshold had to go down from 0.2 to 0.08 and 0.06.

**Why:**
- A derivative compares neighboring pixels, so any small random change gives a big response. In 1.2 it sees the raw pixels, so noise and texture become edges.
- The DoG blurs and differentiates in one step, since (I * G) * Dx = I * (G * Dx). The Gaussian averages neighbors first, so noise and texture are smoothed out before the derivative and only changes that last over a few pixels remain.
- The same averaging spreads each edge over about sigma pixels, so the contours get thicker and the peak gradient drops, which is why the threshold is lower.
- A bigger sigma would remove more detail and make thicker edges, a smaller sigma would be closer to 1.2.