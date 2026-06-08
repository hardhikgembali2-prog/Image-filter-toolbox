import numpy as np
import matplotlib.pyplot as plt

def load_img(path: str) -> np.ndarray:
    """Loads an image, converts to float64 grayscale in range 0.0–1.0."""
    img = plt.imread(path)
    if img is None:
        raise FileNotFoundError(f'File not found at {path}')

    img = img.astype('float64')

    if img.max() > 1.0:
        img = img / 255

    if img.ndim == 3:
        img = 0.2989 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

    return(img)


def convolve(img_path: str , kernel: np.ndarray) -> np.ndarray:
    """Convolves a grayscale image with a given kernel using zero-padding."""
    img = load_img(img_path)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded_img = np.pad(img, ((pad_h, pad_h),(pad_w, pad_w)), mode='constant', constant_values=0)
    kernel = np.flipud(np.fliplr(kernel))
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s = 0.0
            for u in range(kernel.shape[0]):
                for v in range(kernel.shape[1]):
                    s += padded_img[i+u , j+v] * kernel[u, v]

            result[i, j] = s
    result = np.clip(result, 0.0, 1.0)
    return(result)

def k_maker(k_size: int, typ: int, sigma = 1.5) -> np.ndarray:
    """
        Returns a convolution kernel.
        typ 1 = box blur
        typ 2 = gaussian blur (sigma controls spread)
        typ 3 = sharpening
        typ 4 = sobel (returns kx, ky)
        typ 5 = laplacian
        """
    assert typ in [1, 2, 3, 4, 5]
    if typ == 1: #box filter
        k = np.ones((k_size, k_size))
        k = k/k.sum()
        return k

    elif typ == 2: #gaussian blur
        k = np.zeros((k_size,k_size), dtype= 'float64')
        center = k_size//2
        for i in range(k_size):
            for j in range(k_size):
                x = i - center
                y = j - center
                k[i, j] = (1/(2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
        k = k / k.sum()
        return k

    elif typ == 3: #sharpening
        k = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype='float64')
        return k

    elif typ == 4: #sobel filter
        kx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype='float64')

        ky = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype='float64')
        return kx, ky

    elif typ == 5: #laplacian
        k = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype='float64')
        return k

original = load_img('lena.jpg')

box      = convolve('lena.jpg', k_maker(5, 1))
gaussian = convolve('lena.jpg', k_maker(5, 2))
sharp    = convolve('lena.jpg', k_maker(3, 3))
laplace  = convolve('lena.jpg', k_maker(3, 5))

kx, ky   = k_maker(3, 4)
sobel    = np.clip(np.sqrt(convolve('lena.jpg', kx)**2 +
                           convolve('lena.jpg', ky)**2), 0.0, 1.0)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0,0].imshow(original, cmap='gray')
axes[0,0].set_title('Original')

axes[0,1].imshow(box, cmap='gray')
axes[0,1].set_title('Box Blur')

axes[0,2].imshow(gaussian, cmap='gray')
axes[0,2].set_title('Gaussian Blur')

axes[1,0].imshow(sharp, cmap='gray')
axes[1,0].set_title('Sharpening')

axes[1,1].imshow(sobel, cmap='gray')
axes[1,1].set_title('Sobel Edges')

axes[1,2].imshow(laplace, cmap='gray')
axes[1,2].set_title('Laplacian Edges')

for ax in axes.flat:
    ax.axis('off')

plt.suptitle('Image Filter Toolbox — NumPy from scratch', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('comparison_grid.png', bbox_inches='tight')
plt.show()