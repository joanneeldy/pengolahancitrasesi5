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


# Hasil
"""
Jumlah total piksel: 1555200 piksel
Intensitas dominan adalah 175 dengan jumlah 12908 piksel.
Intensitas 0: 52 piksel
Intensitas 1: 187 piksel
Intensitas 2: 538 piksel
Intensitas 3: 794 piksel
Intensitas 4: 1076 piksel
Intensitas 5: 1674 piksel
Intensitas 6: 1817 piksel
Intensitas 7: 2458 piksel
Intensitas 8: 3584 piksel
Intensitas 9: 3740 piksel
Intensitas 10: 4560 piksel
Intensitas 11: 5709 piksel
Intensitas 12: 6628 piksel
Intensitas 13: 7307 piksel
Intensitas 14: 6650 piksel
Intensitas 15: 6007 piksel
Intensitas 16: 6031 piksel
Intensitas 17: 5874 piksel
Intensitas 18: 5837 piksel
Intensitas 19: 5635 piksel
Intensitas 20: 5193 piksel
Intensitas 21: 5203 piksel
Intensitas 22: 5043 piksel
Intensitas 23: 5165 piksel
Intensitas 24: 5150 piksel
Intensitas 25: 4941 piksel
Intensitas 26: 4970 piksel
Intensitas 27: 4911 piksel
Intensitas 28: 4932 piksel
Intensitas 29: 5105 piksel
Intensitas 30: 5272 piksel
Intensitas 31: 5381 piksel
Intensitas 32: 5634 piksel
Intensitas 33: 5976 piksel
Intensitas 34: 6011 piksel
Intensitas 35: 6275 piksel
Intensitas 36: 6482 piksel
Intensitas 37: 6655 piksel
Intensitas 38: 6735 piksel
Intensitas 39: 6748 piksel
Intensitas 40: 6872 piksel
Intensitas 41: 6984 piksel
Intensitas 42: 7263 piksel
Intensitas 43: 7479 piksel
Intensitas 44: 7494 piksel
Intensitas 45: 7464 piksel
Intensitas 46: 7466 piksel
Intensitas 47: 7392 piksel
Intensitas 48: 7294 piksel
Intensitas 49: 7099 piksel
Intensitas 50: 6918 piksel
Intensitas 51: 6830 piksel
Intensitas 52: 6945 piksel
Intensitas 53: 6835 piksel
Intensitas 54: 6612 piksel
Intensitas 55: 6523 piksel
Intensitas 56: 6668 piksel
Intensitas 57: 6544 piksel
Intensitas 58: 6446 piksel
Intensitas 59: 6234 piksel
Intensitas 60: 6133 piksel
Intensitas 61: 5956 piksel
Intensitas 62: 6029 piksel
Intensitas 63: 5976 piksel
Intensitas 64: 5648 piksel
Intensitas 65: 5779 piksel
Intensitas 66: 5538 piksel
Intensitas 67: 5513 piksel
Intensitas 68: 5168 piksel
Intensitas 69: 5212 piksel
Intensitas 70: 5091 piksel
Intensitas 71: 5084 piksel
Intensitas 72: 4986 piksel
Intensitas 73: 4862 piksel
Intensitas 74: 4761 piksel
Intensitas 75: 4586 piksel
Intensitas 76: 4581 piksel
Intensitas 77: 4610 piksel
Intensitas 78: 4827 piksel
Intensitas 79: 4752 piksel
Intensitas 80: 4655 piksel
Intensitas 81: 4792 piksel
Intensitas 82: 4901 piksel
Intensitas 83: 4761 piksel
Intensitas 84: 4834 piksel
Intensitas 85: 4871 piksel
Intensitas 86: 4943 piksel
Intensitas 87: 4833 piksel
Intensitas 88: 4979 piksel
Intensitas 89: 5020 piksel
Intensitas 90: 5244 piksel
Intensitas 91: 5319 piksel
Intensitas 92: 5215 piksel
Intensitas 93: 5354 piksel
Intensitas 94: 5514 piksel
Intensitas 95: 5560 piksel
Intensitas 96: 5715 piksel
Intensitas 97: 5608 piksel
Intensitas 98: 5640 piksel
Intensitas 99: 5745 piksel
Intensitas 100: 5764 piksel
Intensitas 101: 5748 piksel
Intensitas 102: 5884 piksel
Intensitas 103: 6102 piksel
Intensitas 104: 6074 piksel
Intensitas 105: 6166 piksel
Intensitas 106: 6233 piksel
Intensitas 107: 6299 piksel
Intensitas 108: 6432 piksel
Intensitas 109: 6416 piksel
Intensitas 110: 6206 piksel
Intensitas 111: 6342 piksel
Intensitas 112: 6288 piksel
Intensitas 113: 6090 piksel
Intensitas 114: 6091 piksel
Intensitas 115: 6003 piksel
Intensitas 116: 5982 piksel
Intensitas 117: 5913 piksel
Intensitas 118: 5835 piksel
Intensitas 119: 5986 piksel
Intensitas 120: 5712 piksel
Intensitas 121: 5973 piksel
Intensitas 122: 5854 piksel
Intensitas 123: 5873 piksel
Intensitas 124: 5895 piksel
Intensitas 125: 6041 piksel
Intensitas 126: 6043 piksel
Intensitas 127: 6120 piksel
Intensitas 128: 6015 piksel
Intensitas 129: 6028 piksel
Intensitas 130: 5815 piksel
Intensitas 131: 5879 piksel
Intensitas 132: 5910 piksel
Intensitas 133: 5952 piksel
Intensitas 134: 5896 piksel
Intensitas 135: 5710 piksel
Intensitas 136: 5588 piksel
Intensitas 137: 5370 piksel
Intensitas 138: 5304 piksel
Intensitas 139: 5157 piksel
Intensitas 140: 4925 piksel
Intensitas 141: 4768 piksel
Intensitas 142: 4588 piksel
Intensitas 143: 4487 piksel
Intensitas 144: 4264 piksel
Intensitas 145: 4059 piksel
Intensitas 146: 3793 piksel
Intensitas 147: 3936 piksel
Intensitas 148: 6115 piksel
Intensitas 149: 7301 piksel
Intensitas 150: 6628 piksel
Intensitas 151: 6492 piksel
Intensitas 152: 6393 piksel
Intensitas 153: 7145 piksel
Intensitas 154: 10213 piksel
Intensitas 155: 6927 piksel
Intensitas 156: 5716 piksel
Intensitas 157: 5791 piksel
Intensitas 158: 6085 piksel
Intensitas 159: 5978 piksel
Intensitas 160: 6070 piksel
Intensitas 161: 6793 piksel
Intensitas 162: 7528 piksel
Intensitas 163: 7908 piksel
Intensitas 164: 6946 piksel
Intensitas 165: 8122 piksel
Intensitas 166: 7942 piksel
Intensitas 167: 8032 piksel
Intensitas 168: 8185 piksel
Intensitas 169: 9048 piksel
Intensitas 170: 10486 piksel
Intensitas 171: 11252 piksel
Intensitas 172: 10851 piksel
Intensitas 173: 10914 piksel
Intensitas 174: 12249 piksel
Intensitas 175: 12908 piksel
Intensitas 176: 12216 piksel
Intensitas 177: 11844 piksel
Intensitas 178: 12430 piksel
Intensitas 179: 11843 piksel
Intensitas 180: 11472 piksel
Intensitas 181: 11182 piksel
Intensitas 182: 10804 piksel
Intensitas 183: 10382 piksel
Intensitas 184: 10144 piksel
Intensitas 185: 9363 piksel
Intensitas 186: 9200 piksel
Intensitas 187: 10284 piksel
Intensitas 188: 10934 piksel
Intensitas 189: 10090 piksel
Intensitas 190: 9383 piksel
Intensitas 191: 9321 piksel
Intensitas 192: 10756 piksel
Intensitas 193: 11179 piksel
Intensitas 194: 10906 piksel
Intensitas 195: 11144 piksel
Intensitas 196: 12520 piksel
Intensitas 197: 11775 piksel
Intensitas 198: 10310 piksel
Intensitas 199: 9979 piksel
Intensitas 200: 10425 piksel
Intensitas 201: 12368 piksel
Intensitas 202: 12018 piksel
Intensitas 203: 11509 piksel
Intensitas 204: 10605 piksel
Intensitas 205: 9261 piksel
Intensitas 206: 8949 piksel
Intensitas 207: 8798 piksel
Intensitas 208: 7939 piksel
Intensitas 209: 7357 piksel
Intensitas 210: 7627 piksel
Intensitas 211: 7214 piksel
Intensitas 212: 5573 piksel
Intensitas 213: 4926 piksel
Intensitas 214: 5874 piksel
Intensitas 215: 6452 piksel
Intensitas 216: 6796 piksel
Intensitas 217: 6727 piksel
Intensitas 218: 6215 piksel
Intensitas 219: 6591 piksel
Intensitas 220: 6734 piksel
Intensitas 221: 7165 piksel
Intensitas 222: 7808 piksel
Intensitas 223: 8507 piksel
Intensitas 224: 7404 piksel
Intensitas 225: 7958 piksel
Intensitas 226: 7041 piksel
Intensitas 227: 6866 piksel
Intensitas 228: 6325 piksel
Intensitas 229: 6272 piksel
Intensitas 230: 6143 piksel
Intensitas 231: 5573 piksel
Intensitas 232: 4860 piksel
Intensitas 233: 3235 piksel
Intensitas 234: 2573 piksel
Intensitas 235: 1869 piksel
Intensitas 236: 711 piksel
Intensitas 237: 351 piksel
Intensitas 238: 185 piksel
Intensitas 239: 82 piksel
Intensitas 240: 53 piksel
Intensitas 241: 33 piksel
Intensitas 242: 14 piksel
Intensitas 243: 19 piksel
Intensitas 244: 6 piksel
Intensitas 245: 2 piksel
Intensitas 246: 1 piksel
Intensitas 247: 1 piksel
Intensitas 248: 0 piksel
Intensitas 249: 0 piksel
Intensitas 250: 0 piksel
Intensitas 251: 0 piksel
Intensitas 252: 0 piksel
Intensitas 253: 0 piksel
Intensitas 254: 0 piksel
Intensitas 255: 0 piksel
"""