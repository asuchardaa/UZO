from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import numpy as np


def load_image(filepath):
    """
    pouze načte obrázek
    """
    image_data = cv2.imread(filepath)
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


def calculate_histogram(image):
    """
    Výpočet histogramu
    """
    return cv2.calcHist([image], [0], None, [256], [0, 256])


def preprocess_image(filepath):
    """
    Basically preprocessing
    """
    image = load_image(filepath)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    histogram = calculate_histogram(grayscale_image)

    return {
        "histogram": histogram,
        "image": image
    }


def compare_images(data, image_index):
    """
    Funkce pro porovnání histogramů s ostatními
    """
    image_distances = [
        cv2.compareHist(
            data[image_index]["histogram"],
            data[x]["histogram"],
            cv2.HISTCMP_INTERSECT
        ) for x, _ in enumerate(data)
    ]

    return np.array(image_distances)


def plot_images(data, indices, image_index, dimension):
    """
    Funkce pro finální vykreslení :-)
    """
    plt.figure("Porovnání (obrázky)")
    for index in range(len(indices)):
        plt.subplot(dimension, dimension, (image_index * (dimension)) + index + 1)
        plt.imshow(data[indices[index]]["image"])


def main():
    """
    Hlavní funkce
    :return:
    """
    images = ['im01.jpg', 'im02.jpg', 'im03.jpg', 'im04.jpg', 'im05.jpg', 'im06.jpg', 'im07.jpg', 'im08.jpg', 'im09.jpg']

    # zde nejdriv nactu, pak nasleduje konverze do grayscale, a teprve z toho muzu vypocitat histrogram
    data = [preprocess_image(Path(x).as_posix()) for x in images]
    dimension = len(data)

    # porovnavam kazdy s kazdym a vykreslim histogram
    for image_index, _ in enumerate(data):
        image_distances = compare_images(data, image_index)
        indices = np.argsort(image_distances)[::-1]

        print(image_index, image_distances[indices])
        plot_images(data, indices, image_index, dimension)

    plt.show()


if __name__ == "__main__":
    main()
    plt.close('all')
