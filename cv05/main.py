# Autor: Adam Sucharda, Marcel Horváth 2024
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path


def apply_filter(image, kernel):
    """
    Metoda pro aplikaci filtru na obrázek se zadaným upraveným jádrem (průměrování)
    :param image:
    :param kernel:
    :return:
    """
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


def get_low_variance_index(masks):
    """
    Metoda pro získání indexu masky s nejnižší variancí z celého listu (masky)
    :param masks:
    :return:
    """
    variances = [np.var(mask) for mask in masks]
    return np.argmin(variances)


def average_with_rotating_mask(image, size=3):
    """
    Metoda, která mi aplikuje filtr na obrázek (rotující maska)
    Hledám nejnižší varianci v okolí pixelu a průměruji hodnoty
    :param image:
    :param size:
    :return:
    """

    # abych mel kde ukladat
    copy = np.zeros(image.shape)

    # jedu pred kazdej pixel v obrazku, aby se aplikovalo vsude
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            masks = []
            # jedu i pres okolni pixely, aby se maska mohla spravne aplikovat
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    # vypocet souradnic masky
                    top = max(y + dy - size // 2, 0)
                    bottom = min(y + dy + size // 2 + 1, image.shape[0])
                    left = max(x + dx - size // 2, 0)
                    right = min(x + dx + size // 2 + 1, image.shape[1])

                    mask = image[top:bottom, left:right]
                    masks.append(mask)
            # ziskani indexu masky s nejnizsi varianci
            lowest_variance_index = get_low_variance_index(masks)
            # prirazeni prumeru masky do kopie pro lepsi vysledne zobrazeni
            copy[y, x] = np.mean(masks[lowest_variance_index])

    return copy


def plot_images_with_histogram_spectrum(original, result, title, output_path):
    """
    Metoda pro zobrazení výsledku
    :param original:
    :param result:
    :param title:
    :param output_path:
    :return:
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, 2)
    plt.hist(original.ravel(), 256, [0, 256])
    plt.title('Histogram (Original)')
    plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, 3)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(original)))), cmap='jet')
    plt.title('Spectrum (Original)')
    plt.colorbar()
    plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, 4)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, 5)
    plt.hist(result.ravel(), 256, [0, 256])
    plt.title('Histogram (Filtered)')
    plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, 6)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(result)))), cmap='jet')
    plt.title('Spectrum (Filtered)')
    plt.colorbar()
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()

    output_filename = f"{title.replace(' ', '_').lower()}.png"
    output_file_path = os.path.join(output_path, output_filename)
    plt.savefig(output_file_path)
    plt.show()


def main():
    """
    Hlavní metoda programu
    :return:
    """
    #file = "cv05_robotS.bmp"
    file = "cv05_PSS.bmp"
    bgr_img = cv2.imread(Path(file).as_posix())
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    output_dir = "znak"
    os.makedirs(output_dir, exist_ok=True)
    original_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_img))))

    # Averaging
    kernel = 1 / 9 * np.ones((3, 3))
    mean_filtered = apply_filter(gray_img, kernel)
    plot_images_with_histogram_spectrum(gray_img, mean_filtered, 'Average', output_dir)

    # Rotation Mask
    rotation_filtered = average_with_rotating_mask(gray_img)
    plot_images_with_histogram_spectrum(gray_img, rotation_filtered, 'Average, rotation mask', output_dir)

    # Median
    median_filtered = cv2.medianBlur(gray_img, 3)
    plot_images_with_histogram_spectrum(gray_img, median_filtered, 'Median', output_dir)


if __name__ == "__main__":
    "Vim co delam, takze tohle je v pohode."
    main()
