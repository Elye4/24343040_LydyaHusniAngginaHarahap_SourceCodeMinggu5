import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# LOAD GAMBAR
# =========================
image = cv2.imread('cameramen.jpg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =========================
# TAMBAH NOISE
# =========================
def add_gaussian_noise(img):
    noise = np.random.normal(0, 25, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(img):
    noisy = img.copy()
    prob = 0.02
    rnd = np.random.rand(*img.shape)
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def add_speckle(img):
    noise = np.random.randn(*img.shape)
    noisy = img + img * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

gaussian = add_gaussian_noise(image)
sp = add_salt_pepper(image)
speckle = add_speckle(image)

# =========================
# FILTER
# =========================
def mean_filter(img, k):
    return cv2.blur(img, (k, k))

def gaussian_filter(img, sigma):
    return cv2.GaussianBlur(img, (5,5), sigma)

def median_filter(img, k):
    return cv2.medianBlur(img, k)

def max_filter(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(img, kernel)

# =========================
# SSIM MANUAL
# =========================
def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = 6.5025
    C2 = 58.5225

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1**2, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11,11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1*img2, (11,11), 1.5) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# =========================
# EVALUASI
# =========================
def evaluate(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    psnr = cv2.PSNR(original, filtered)
    ssim_val = calculate_ssim(original, filtered)
    return mse, psnr, ssim_val

# =========================
# PROSES SEMUA
# =========================
results = []

noises = {
    "Gaussian": gaussian,
    "SaltPepper": sp,
    "Speckle": speckle
}

filters = {
    "Mean3": lambda x: mean_filter(x, 3),
    "Mean5": lambda x: mean_filter(x, 5),
    "Gaussian1": lambda x: gaussian_filter(x, 1),
    "Gaussian2": lambda x: gaussian_filter(x, 2),
    "Median3": lambda x: median_filter(x, 3),
    "Median5": lambda x: median_filter(x, 5),
    "Max": lambda x: max_filter(x)
}

for noise_name, noisy_img in noises.items():
    for filter_name, func in filters.items():
        start = time.time()
        filtered = func(noisy_img)
        end = time.time()

        mse, psnr, ssim_val = evaluate(image, filtered)

        results.append([noise_name, filter_name, mse, psnr, ssim_val, end-start])

        # Simpan gambar
        cv2.imwrite(f"{noise_name}_{filter_name}.jpg", filtered)

# =========================
# PRINT HASIL
# =========================
print("\n=== HASIL EVALUASI ===")
for r in results:
    print(f"Noise: {r[0]} | Filter: {r[1]} | MSE: {r[2]:.2f} | PSNR: {r[3]:.2f} | SSIM: {r[4]:.4f} | Time: {r[5]:.4f}")

# =========================
# VISUALISASI
# =========================
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.title("Citra Asli")
plt.imshow(image, cmap='gray')

plt.subplot(2,2,2)
plt.title("Gaussian Noise")
plt.imshow(gaussian, cmap='gray')

plt.subplot(2,2,3)
plt.title("Salt & Pepper")
plt.imshow(sp, cmap='gray')

plt.subplot(2,2,4)
plt.title("Speckle Noise")
plt.imshow(speckle, cmap='gray')

plt.show()