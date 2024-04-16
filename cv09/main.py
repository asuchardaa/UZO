# Autoři: Marcel Horváth, Adam Sucharda @ 2024

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_top_hat(image, kernel_size=10):
    """
    Funkce pro aplikaci top-hat transformace na obrázek
    :param image:
    :param kernel_size:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    top_hat_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return top_hat_img


def apply_threshold(image, threshold):
    """
    Funkce pro aplikaci prahování na obrázek
    :param image:
    :param threshold:
    :return:
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def getHistogram(image):
    """
    Funkce pro výpočet histogramu obrázku
    :param image:
    :return:
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    normalized_histogram = histogram / (image.shape[0] * image.shape[1])
    return normalized_histogram


def label_objects(binary_img):
    """
    Funkce pro označení objektů na obrázku
    :param binary_img:
    :return:
    """
    num_labels, labels = cv2.connectedComponents(binary_img)
    return labels, num_labels - 1


def filter_small_objects(labels, min_size=50):
    """
    Funkce pro filtrování malých objektů na obrázku
    :param labels:
    :param min_size:
    :return:
    """
    num_labels = np.max(labels)
    label_sizes = np.bincount(labels.ravel())
    filtered_labels = labels.copy()
    for i in range(1, num_labels + 1):
        if label_sizes[i] < min_size:
            filtered_labels[labels == i] = 0
    return filtered_labels, len(np.unique(filtered_labels)) - 1


def calculate_centroids(labels):
    """
    Funkce pro výpočet těžišť objektů na obrázku
    :param labels:
    :return:
    """
    centroids = []
    for label in range(1, np.max(labels) + 1):
        ys, xs = np.where(labels == label)
        if xs.size > 0 and ys.size > 0:
            centroid = (int(np.mean(xs)), int(np.mean(ys)))
            centroids.append(centroid)
    return centroids


def draw_centroids(image, centroids):
    """
    Funkce pro vykreslení těžiště do zrnka rýže na obrázku
    :param image:
    :param centroids:
    :return:
    """
    image = image.copy()
    for x, y in centroids:
        cv2.drawMarker(image, (x, y), (255, 0, 0), markerType=cv2.MARKER_STAR, markerSize=5)
    return image


def main():
    """
    Hlavní řídící funkce programu
    :return:
    """
    image_path = 'cv09_rice.bmp'
    image = cv2.imread(image_path, 0)
    top_hat_image = apply_top_hat(image)

    threshold_value = 60
    binary_original = apply_threshold(image, 120)
    binary_top_hat = apply_threshold(top_hat_image, threshold_value)

    plt.figure(figsize=(10, 7))
    plt.subplot(221), plt.plot(getHistogram(image)), plt.title('Original Histogram')
    plt.subplot(222), plt.imshow(binary_original, cmap='gray'), plt.title('Original Binary')
    plt.subplot(223), plt.plot(getHistogram(top_hat_image)), plt.title('Top-Hat Histogram')
    plt.subplot(224), plt.imshow(binary_top_hat, cmap='gray'), plt.title('Top-Hat Binary')
    plt.tight_layout()
    plt.show()

    labels_original, num_original = label_objects(binary_original)
    labels_top_hat, num_top_hat = label_objects(binary_top_hat)

    filtered_labels_original, final_count_original = filter_small_objects(labels_original)
    filtered_labels_top_hat, final_count_top_hat = filter_small_objects(labels_top_hat)

    centroids_original = calculate_centroids(filtered_labels_original)
    centroids_top_hat = calculate_centroids(filtered_labels_top_hat)
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    centroid_image = draw_centroids(image_colored, centroids_original + centroids_top_hat)

    plt.imshow(centroid_image), plt.title('Pocet zrnicek:' + str(final_count_top_hat))
    plt.show()

    print(f'Počet zrníček rýže na obrázku: {final_count_top_hat}')


if __name__ == '__main__':
    main()
