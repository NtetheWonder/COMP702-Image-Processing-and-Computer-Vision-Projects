import numpy as np
import cv2

def mean_filter(image, x, y):
    kernel = np.ones((3,3)) / 9.0
    return np.sum(image[x-1:x+2, y-1:y+2] * kernel)

def median_filter(image, x, y):
    return np.median(image[x-1:x+2, y-1:y+2])

def laplacian_filter(image, x, y):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return np.sum(image[x-1:x+2, y-1:y+2] * kernel)

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def gradient_magnitude(image, x, y):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gx = np.sum(image[x-1:x+2, y-1:y+2] * sobel_x)
    gy = np.sum(image[x-1:x+2, y-1:y+2] * sobel_y)
    
    return np.sqrt(gx**2 + gy**2)

def histogram(image):
    hist, _ = np.histogram(image, bins=8, range=(0, 7))  # 3-bit image (0-7 values)
    return hist

def histogram_equalization(image, x, y):
    hist, bins = np.histogram(image.flatten(), bins=8, range=(0,7))
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 7 / (cdf.max() - cdf.min())
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)
    return equalized_image[x, y]

def gamma_correction(image, x, y, gamma=2.5):
    return np.power(image[x, y], gamma)

# Example 5x5 image (3-bit per pixel values between 0-7)
image = np.array([
    [3, 3, 2, 1, 0],
    [2, 3, 4, 2, 1],
    [1, 2, 5, 3, 2],
    [0, 1, 3, 2, 1],
    [1, 0, 2, 3, 4]
])

# Computing results
x, y = 2, 2
print("Mean Filter Output:", mean_filter(image, x, y))
print("Median Filter Output:", median_filter(image, x, y))
print("Laplacian Filter Output:", laplacian_filter(image, x, y))
print("Euclidean Distance (2,2) to (4,3):", euclidean_distance(2, 2, 4, 3))
print("Gradient Magnitude:", gradient_magnitude(image, x, y))
print("Histogram:", histogram(image))
print("Histogram Equalization at (2,2):", histogram_equalization(image, x, y))
print("Gamma Correction at (2,2):", gamma_correction(image, x, y))
