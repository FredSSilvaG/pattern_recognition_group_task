import numpy as np
import cv2
from skimage.transform import resize
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean



# def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """
#     Euclidean distance: √(∑ (a - b)²)
#     """
#     return np.sqrt(np.sum((a - b) ** 2, axis=-1))


# def compute_distance_matrix(features1, features2):
#     n = len(features1)
#     m = len(features2)
#     distance_matrix = np.zeros((n, m))
#     for i in range(n):
#         for j in range(m):
#             distance_matrix[i, j] = euclidean_distance(features1[i], features2[j])
#     return distance_matrix


# def dtw(distance_matrix):
#     n, m = distance_matrix.shape
#     cost = np.zeros((n, m))
#     cost[0, 0] = distance_matrix[0, 0]
#     for i in range(1, n):
#         cost[i, 0] = cost[i-1, 0] + distance_matrix[i, 0]
#     for j in range(1, m):
#         cost[0, j] = cost[0, j-1] + distance_matrix[0, j]
#     for i in range(1, n):
#         for j in range(1, m):
#             cost[i, j] = distance_matrix[i, j] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
#     return cost[-1, -1]

def dtw_distance(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance

# Normalisation
def z_normalize(seq):
    seq = np.array(seq)
    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0)
    std[std == 0] = 1
    return (seq - mean) / std

def binarize_image(img):
    # Convert the image to grayscale if it's not already
    if len(img.shape) == 3:  # Check if the image has 3 channels (color image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure the image is uint8 type
    if img.dtype != np.uint8:
        img = np.uint8(img)  # Convert to 8-bit unsigned integer
     
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 10)


def extract_hog_sequence(image, fixed_size=(100, 100)):
    if image is None or image.size == 0:
        print("Empty image in extract_features")
        return None
    img = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)
   


    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    h, w = binary.shape
    if h == 0 or w == 0:
        print("Invalid image shape:", image.shape)
        return None

    features = []
    for x in range(w):
        col = binary[:, x]

        black_pixels = np.where(col > 0)[0]
        n_black = len(black_pixels)
        n_total = h

        # UC and LC
        if n_black > 0:
            UC = black_pixels[0]
            LC = black_pixels[-1]
        else:
            UC = 0
            LC = h - 1

        # Transitions
        transitions = np.count_nonzero(col[1:] != col[:-1])

        # Black fraction
        frac_black = n_black / n_total

        # Black fraction between UC and LC
        vertical_band = col[UC:LC+1] if UC <= LC else []
        frac_black_band = np.count_nonzero(vertical_band) / max(1, LC - UC + 1)

        features.append([UC, LC, transitions, frac_black, frac_black_band])

    # Convert to array
    features = np.array(features)

    # Compute UC/LC gradients
    uc_grad = np.diff(features[:, 0], append=features[-1, 0])
    lc_grad = np.diff(features[:, 1], append=features[-1, 1])

    # Add gradients to feature vector
    features = np.hstack([features, uc_grad[:, None], lc_grad[:, None]])

    # normalize
    features = z_normalize(features)

    return features
    
