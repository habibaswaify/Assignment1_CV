# Assignment1_CV

## Part 3: Seam Carving
### 3.1 Finding and Removing Seams
A seam is a connected path of pixels on an image vertically or horizontally, where the optimal seam is defined by an energy function. 
#### Methodology:
Finding seams uses dynamic programming by default. This is because the brute force approach has a time complexity of O(3^M) for an M x N image, where 3 is the number of choices: left, center and right. With such complexity, it would not even be possible to perform the coomputation on a computer.
On the other hand, using dynamic programming reduces the cost to O(M x N).

Initially, *broadway_tower.png* is used as an example to visualize the energy computation (which is set to the gradient magnitude by default), visualize the highlighted seams and the reduced image output

![broadway_tower](starter_pack/images/part3_seams/outputs/output_3_1.png)

### 3.2 Results and Comparisons
#### First Comparison: Reduction Method
Seam carving's outputs are measured against those of resizing and cropping respectively, with a constant reduction ratio of 30%.
It is clearly observed that in frequent cases, seam carving is the most accurate for preserving image features, where cropping clips the images' borders thus discarding features blindly and resizing clearly distorts the output. Use *broadway_tower.png* and *1665_Girl_with_a_Pearl_Earring.jpg* as an example.
![broadway_tower_2](starter_pack/images/part3_seams/outputs/output_3_2_3.png)
![girl_with_a_pearl_earring](starter_pack/images/part3_seams/outputs/output_3_2_1.png)

However, seam carving has been shown to be inaccurate when applied to condensed images (Avidan & Shamir, 2007), as shown in *snow.png*. In addition, this inaccuracy is also prominent when applied to flat-texture images, such as *blue_fish.png*.
![snow](starter_pack/images/part3_seams/outputs/output_3_2_4.png)
![blue_fish](starter_pack/images/part3_seams/outputs/output_3_2_2.png)

#### Second Comparison: Energy Functions
Three energy computation functions were tested for seam detection:
* Gradient Magnitude
* Finite Differences
* Derivative of Gaussian (DoG)

Gradient magnitude's and finite differences' seams were almost identical, while DoG's seams showed slight difference, and even a minor distortion in fine details: such as discontinuities. This is primarily due to the blurring effect of DoG before computing the energy. The discontinuity effect is prominent in *starry_night.jpg* and *rocket.jpg*
![starry_night](starter_pack/images/part3_seams/outputs/output_3_2_5.png)
![rocket](starter_pack/images/part3_seams/outputs/output_3_2_6.png)

---
## Bonus
### Object Removal
The user selects an object to remove using `matplotlib` widget while keeping the original image size. The widget creates a resizeable elliptical mask surrounding the object. Seam carving is used to reduce the discontinuities caused by such process. The implementation is found in the `remove_object()` function and it is demonstrated briefly as follows: 
- The image is copied into `working_img`
- Cheapest seams are removed iteratively from the mask and `working_img`
- `working_img`is copied into `seam_copy`, then the cheapest seams of the current `seam_copy` are used to generate new pixels by averaging and they are removed from `seam_copy` afterwards

The copy (`seam_copy`) is used to find insertion locations and it is shrunk every iteration to discard used seams. Choosing the cheapest seam from the same image (`working_img`) and appending it iteratively would affect the image's energy calculation based on synthetic pixels and not the original pixels which are present in the reduced image copy.
The test performed on `broadway_tower.png` is shown below
![tower_mask](starter_pack/images/part3_seams/outputs/bonus_1.png)
![tower_removed](starter_pack/images/part3_seams/outputs/bonus_2.png)