# Histograms of Oriented Gradients for Human Detection

Franco Meng & Laine Mulvay  
**Date:** 23-May-2025

This report aims to present a clear methodology and findings on using Histogram of Oriented Gradients (HOG) and linear SVM for image classification.

## Phase 1 - Dataset Collection

### 1.1 Human Data

The human image data used in this project were sourced from two subfolders, PRID and MIT, within the Pedestrian Attribute Recognition at Far Distance, [PETA (PEdesTrian Attribute) dataset](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html).

To ensure diversity in the dataset, we selected images that represent all viewing angles — front, back, and side. The PRID subset includes multiple images of the same individual from different perspectives, labelled with suffixes such as -a and -b. To maximise training data diversity, we aimed to excluded images of the same person taken from different angles.

The selected human images are stored in either **JPG** or **PNG** formats and have a uniform resolution of **64×128** pixels.

![perfect_human_1](https://github.com/user-attachments/assets/6a74b0a9-55ba-40a9-a68d-71d963d319e4)
![perfect_human_2](https://github.com/user-attachments/assets/6de0d0d4-4511-4496-be58-e78c7585f1a1)


### 1.2 Non-human Data

Non-human images were derived from the [INRIA Person Dataset](https://www.kaggle.com/datasets/jcoral02/inriaperson), which contains 1,811 images along with XML annotations that mark human regions.

To generate negative (non-human) samples:

- We first located the annotated human regions.
- We then extracted horizontally adjacent areas that did not contain a human annotation.

The above pre-processing approach by extracting non-human regions nearby to human regions, created more realistic non-human examples, closely resembling real-world scenes.

All non-human images are in **JPG** format and resized to **64×128** pixels.

![perfect_non_human_2](https://github.com/user-attachments/assets/633e7430-aba8-44d7-bb02-64605aa79559)

![perfect_non_human_1](https://github.com/user-attachments/assets/18c4fa46-288f-46bf-8bf7-3361ef30954f)

### 2.1 Custom Implementation of HOG Feature Descriptor

Instead of using prebuilt functions like `cv2.HOGDescriptor` or `skimage.feature.hog`, we implemented the HOG feature descriptor from scratch. This decision was made to:

- Gain a deeper understanding of the HOG algorithm and its step-by-step process.
- Allow full flexibility in tuning parameters and conducting ablation studies.
- Overcome limitations of existing libraries, such as the `cv2.HOGDescriptor`'s inability to customise filters in OpenCV's implementation - making it unsuitable for experiments involving different filters like Sobel, Prewitt.

### 2.2 Function Definition

```python
compute_hog(image, cell_size=8, block_size=16, num_bins=9, block_stride=1, filter_="default", angle_=180)
```

Parameters:

- `image`: A grayscale image of size 64×128 pixels. Input must be a 2D array.
- `cell_size` (default = 8): Specifies the width and height (in pixels) of each cell. A cell of size 8 means each cell covers an 8×8 pixel region.
- `block_size` (default = 16): Defines the size of a block, measured in pixels. A block of size 16 includes four 8×8 cells (2×2 grid).
- `num_bins` (default = 9): Number of orientation bins in the histogram. If the angle range is 360°, each bin represents a 40° segment.
- `block_stride` (default = 1): The stride when moving the block window, measured in cells. A stride of 1 moves the block by one cell at a time, creating overlapping blocks.
- `filter_` (default="default"): Specifies the filter applied before computing gradient magnitudes and orientations.
  - 'Default': A basic 1D derivative filter [-1, 0, 1] applied in both x and y directions.
  - 'Prewitt': Equal-weight filter that enhances sharp edges.
  - 'Sobel': Weighted filter that emphasises the central pixel, offering smoother gradients and improved noise resistance.
- `angle_` (default=180): The unit is in degrees, not radians. If set to 180°, angles wrap (e.g., 270° becomes 90°). Magnitudes and angles are computed using OpenCV's [cv2.cartToPolar](https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#carttopolar), which provides angle estimates with ~0.3° precision.





### 2.3 HOG Feature Extraction Pipeline

1. Raw Image Input
2. Grayscale Conversion
3. Resizing to a fixed dimension of 64×128 pixels.
4. Gradient Filtering
5. Gradient Magnitude and Orientation Calculation
6. Histogram of Oriented Gradients (HOG) Construction
7. Block Normalisation and Histogram Concatenation

   - Histograms from cells are grouped into overlapping blocks (e.g., 2×2 cells or 16×16 pixels).
   - Within each block, histograms are concatenated and normalised to improve invariance to lighting changes.
8. Final Feature Vector Creation

   - All block-level normalised histograms are concatenated into a single feature vector for the image.
   - The length of the final feature vector depends on parameters like block size, cell size, and block stride.

At this stage, each image has been transformed into a numeric feature representation, ready for feeding into a Linear SVM model.

### 2.4 SVM and Classification

After labelling the corresponding features with 1 and 0, we use `LinearSVC()` from scikit-learn for model training, with no advanced hyperparameter tuning.

### 2.5 Evaluation

We evaluated the model using two complementary methods:

- Standard ROC Curve and AUC Score
- Detection Error Tradeoff (DET) curve.

The DET curve plots miss rate (1-TPR) vs False Positives Per Window (FFPW).

This evaluation method was adopted from the original HOG paper, where detection was performed across sliding windows on larger images. But in our case, each window equals the entire image, so FPPW is equivalent to FP.

The DET curve is plotted on a logarithmic scale, allowing better insight into small performance differences.


# Phase 3 - Ablation Study

Due to the high performance (AUC close to 1) achieved on the perfect testing dataset, we conducted an ablation study using the imperfect testing dataset to more thoroughly investigate the model's sensitivity under different parameters.

In this ablation study, rather than performing a full grid search across all possible parameter combinations, we strategically focused on two specific sets of parameters. This selective approach allows us to isolate and analyse the impact of individual factors.

For each combination, the HOG features were re-extracted from the training and testing datasets, and the model was retrained from scratch using the same SVM settings.

The resulting performance was tabled using both Accuracy and AUC scores on the imperfect testing dataset.

We mainly focus on the AUC scores. The comparison plots below present how different configurations impact the model's ability to generalise.

### 3.1 Set 1: Cell size & Block size

We examined the following 10 cell size and block size combinations:

**[(2,4), (2,6), (2,8), (2,16), (4,8), (4,16), (4,32), (8,16), (8,32), (16,32)]**

![cell_block_ablation](https://github.com/user-attachments/assets/d683ab40-5a32-4285-b44d-b3e26c81f95d)



### 3.2 Set 2: Bin size and Orientation Angle

We examined the following 9 bin size and orientation angle combinations:

**[(3,180), (4,180), (6,180), (8,180), (9,180), (12,180), (6,360), (8,360), (12,360), (18,360)]**


![bin_angle_ablation](https://github.com/user-attachments/assets/d13332d7-6871-46c2-a235-8e12e44a54c8)

###  3.3 Final HOG Parameters and Model

After conducting the ablation studies, the following parameters were selected for the final HOG descriptor and SVM model:

| Parameter      | Final Value | Default Value |
|----------------|-------------|---------------|
| Cell size      | **4 x 4**       | 8 x 8         |
| Block Size     | **32 x 32**     | 16 x 16       |
| Bin Number     | **9**           | 9             |
| Orientation Angle | **180°**     | 180°          |
| Filter         | **[-1, 0, 1]**  | [-1, 0, 1]    |
| Block Stride   | **1**           | 1             |

It is worth noting that an earlier comparison of different gradient filters was also conducted, although not included in the final report. In that initial study, the simple [-1,0,1] filter without any smoothing yielded the best performance. Dalal and Triggs (2005) offered some explanation, stating that excessive smoothing before gradient computation can damage HOG performance, highlighting the importance of preserving information retrieved from **abrupt edges at fine scales**.


![evaluate__PETA_INRIA_h250p_nh250pp_c4_b32_n9_s1_180_unperfect_200](https://github.com/user-attachments/assets/0c0d2a68-7a02-417f-975d-91499ca8e2d6)


![evaluate__PETA_INRIA_h250p_nh250pp_c8_b16_n9_s1_default_180_perfect_200](https://github.com/user-attachments/assets/1d2f948f-6853-4b31-9ce4-28f8db3fba19)

![evaluate__PETA_INRIA_h250p_nh250pp_c8_b16_n9_s1_default_180_unperfect_200_misclassified](https://github.com/user-attachments/assets/0de5eb07-f820-415d-8fc9-2b79681de0b8)


