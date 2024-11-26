"""
Tugas Pengolahan Citra Sesi 5
"""

# Import libraries
import numpy as np
import imageio as img
import matplotlib.pyplot as plt

# Load image
image = img.imread('R.jpg')

# Convert RGB to Grayscale using luminosity method
grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Save the grayscale image
img.imwrite('gray_image.jpg', grayscale)

# Calculate histogram
histogram, bin_edges = np.histogram(grayscale, bins=256, range=(0, 255))

# Plot grayscale image
plt.figure(figsize=(6, 6))
plt.title("Grayscale Image")
plt.imshow(grayscale, cmap='gray')
plt.axis('off')
plt.show()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], histogram, width=1, edgecolor="black", color="gray")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Histogram of Grayscale Image")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()

# Total pixels per intensity
total_pixels = np.sum(histogram)
dominant_intensity = np.argmax(histogram)
print(f"Jumlah total piksel: {total_pixels} piksel")
print(
    f"Intensitas dominan adalah {dominant_intensity} dengan jumlah "
    f"{histogram[dominant_intensity]} piksel."
)

# Print detailed pixel counts for each intensity
for intensity, count in enumerate(histogram):
    print(f"Intensitas {intensity}: {count} piksel")
    