import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
from matplotlib import pyplot as plt


# --- Image Classes ---
@dataclass(frozen=True)
class Image:
    data: np.ndarray

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def width(self):
        return self.data.shape[1]


class ColorImage(Image):
    def toGrayscale(self):
        return GrayscaleImage(cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY))


class GrayscaleImage(Image):
    def toBinary(self, threshold: Optional[int] = None, inverse: bool = True):
        if threshold is None:
            _, binaryData = cv2.threshold(self.data, 0, 1, cv2.THRESH_OTSU)
            if inverse:
                binaryData = np.invert(binaryData.astype('bool')).astype('uint8')
        else:
            if inverse:
                _, binaryData = cv2.threshold(self.data, threshold, 1, cv2.THRESH_BINARY_INV)
            else:
                _, binaryData = cv2.threshold(self.data, threshold, 1, cv2.THRESH_BINARY)
        return BinaryImage(binaryData)

    def whitePointAdjusted(self, strength: float = 1.0):
        hist, _ = np.histogram(self.data, 255, range=(0, 255))
        whitePoint = np.argmax(hist)
        whiteScaleFactor = 255 / whitePoint * strength
        return GrayscaleImage(cv2.addWeighted(self.data, whiteScaleFactor, self.data, 0, 0))


class BinaryImage(Image):
    def toColor(self):
        return ColorImage(cv2.cvtColor(self.data * 255, cv2.COLOR_GRAY2BGR))


# --- Vision (Morphological Ops) ---
def openImage(binaryData: np.ndarray) -> np.ndarray:
    element = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))
    eroded = cv2.erode(binaryData, element)
    opened = cv2.dilate(eroded, element)
    return opened


# --- Grid Detection Logic (Kernel Approach) ---
def kernelApproach(colorImage: ColorImage) -> BinaryImage:
    grayscale = colorImage.toGrayscale()
    binaryImage = grayscale.toBinary(threshold=240)  # Initial binarization to separate background

    opened = openImage(binaryImage.data)
    opened = openImage(opened)  # Apply opening twice to enhance grid

    subtracted = cv2.subtract(binaryImage.data, opened)

    final = cv2.erode(subtracted, cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)))
    return BinaryImage(final)


# --- Utility ---
def readImage(path: Union[str, Path]) -> ColorImage:
    data = cv2.imread(str(path))
    return ColorImage(data)


def saveImage(image: Image, path: Union[str, Path]):
    if isinstance(image, BinaryImage):
        img = image.toColor().data
    else:
        img = image.data
    cv2.imwrite(str(path), img)


def showComparison(original: Image, processed: Image, titles=("Original", "Grid Detected"), save_path="comparison_plot.png"):
    plt.figure(figsize=(12, 6))

    # Original
    plt.subplot(1, 2, 1)
    if len(original.data.shape) == 2:
        plt.imshow(original.data, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(original.data, cv2.COLOR_BGR2RGB))
    plt.title(titles[0])
    plt.axis('off')

    # Processed
    plt.subplot(1, 2, 2)
    if len(processed.data.shape) == 2:
        plt.imshow(processed.data, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(processed.data, cv2.COLOR_BGR2RGB))
    plt.title(titles[1])
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Comparison plot saved to: {save_path}")


# --- MAIN ---
if __name__ == "__main__":
    input_path = "output_cropped3.png"
    # output_path = "grid_detected.png"
    comparison_path = "comparison_plot3.png"

    img = readImage(input_path)

    print("Applying kernel approach...")
    grid_img = kernelApproach(img)

    # print("Saving individual output...")
    # saveImage(grid_img, output_path)
    # print(f"Saved: {output_path}")

    print("Showing side-by-side comparison...")
    showComparison(img, grid_img, titles=("Original Image", "Grid Detected"), save_path=comparison_path)
