import numpy as np
import cv2
from matplotlib import pyplot as plt


def segment_objects(img):
    """
    Funkce pro segmentaci objektů na obrázku
    :param img: Vstupní obrázek
    :return: Segmenovaný obrázek, středy nalezených objektů, seznam velikostí objektů
    """
    img_copy = np.copy(img)

    img_copy, neighbors = first_iteration(img_copy)

    areas, img_copy = second_iteration(img_copy, neighbors)

    centers_array = calculate_centers(img_copy, areas)

    return img_copy, centers_array, list(areas)


def calculate_centers(img, areas):
    """
    Funkce pro výpočet těžišť objektů
    :param img: Segmentovaný obrázek
    :param areas: Seznam identifikovaných objektů
    :return: Seznam středů objektů
    """
    centers_of_objects = []
    for area in areas:
        # pro kazdej objekt musim vypocitat teziste, jinak to nesouhlasi s testem (1,5)
        center = (int(general_moment(img, 0, 1, area) / general_moment(img, 0, 0, area)),
                  int(general_moment(img, 1, 0, area) / general_moment(img, 0, 0, area)))
        centers_of_objects.append(center)
    return centers_of_objects


def first_iteration(img):
    """
    První iterace při segmentaci objektů
    :param img: Vstupní obrázek
    :return: Segmentovaný obrázek, sousedé
    """
    counter = 1
    neighbors = {}

    # Zacinam loopvat pres kazdej pixel v obrazku
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1] - 1):
            # basic kontrola, jenom kdyby nahodou byl pixel mimo obrazek -> hazelo mi to error
            if img[i, j] != 1:
                continue
            # okolni pixely okolo aktualniho pixelu
            pos = np.array([img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1], img[i, j - 1]])
            # kontrola pixelu zda 0 nebo 1
            if sum(pos) == 0:
                counter += 1
                img[i, j] = counter
                neighbors[counter] = set()
                continue
            # Okolni pixel ma nenulovou hodnotu -> pridame do existujiciho
            nzero = np.any(pos)
            # kontroluju non zero pixely
            if nzero:
                areas = np.argwhere(pos > 1)
                if len(areas) == 1:
                    img[i, j] = pos[areas[0][0]]
                else:
                    k = areas[0][0]
                    img[i, j] = pos[k]
                    for x in areas:
                        neighbors[pos[k]].add(pos[x][0])

    return img, neighbors


def second_iteration(img, neighbors):
    """
    Druhá iterace při segmentaci objektů
    :param img: Vstupní obrázek
    :param neighbors: Sousedi
    :return: Seznam objektů, segmentovaný obrázek
    """
    # naopak
    for key in reversed(neighbors.keys()):
        # nahrazuju pixely v obrazku
        for area in neighbors[key]:
            img[img == area] = key
    areas = set(img.astype('int').flatten())
    areas.remove(0)

    return areas, img


def general_moment(img, p, q, color):
    """
    Obecný moment pro výpočet těžišť objektů
    :param img: Vstupní obrázek
    :param p: Stupen x
    :param q: Stupen y
    :param color: Identifikace barvy objektu
    :return: Obecný moment
    """
    sum = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # obecna formula pro vypocet obecneho momentu, hodi se nam pro pozdejsi znaceni teziste
            sum += np.power(x, p) * np.power(y, q) * (img[x, y] == color)
    return sum


def classify_coins(seg, centers, areas):
    """
    Klasifikace mincí podle velikosti objektů
    :param seg: Segmentovaný obrázek
    :param centers: Seznam středů objektů
    :param areas: Seznam identifikovaných objektů
    :return: Seznam klasifikací
    """
    classifications = [1] * len(centers)
    for (k, (center, area)) in enumerate(zip(centers, areas)):
        pixel_count = np.count_nonzero(seg == area)
        # dle zadani, nad 4000 pixelu je 5 koruna, jinak koruna
        if pixel_count > 4000:
            classifications[k] = 5
    return classifications


def apply_threshold(img, threshold):
    """
    Aplikace prahování na obrázek
    :param img: Vstupní obrázek
    :param threshold: Hodnota prahu
    :return: Segmentovaný obrázek
    """
    res = np.ones_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] >= threshold:
                res[i, j] = 0
    return res


def plot_segmentation(segmentation, color, centers, file):
    """
    Vykreslení výsledků segmentace
    :param segmentation: Segmentovaný obrázek
    :param color: Barva segmentace
    :param centers: Seznam středů objektů
    :param file: Vstupní obrázek
    :return: None
    """
    plt.subplot(1, 2, 1)
    plt.imshow(segmentation, cmap='gray')
    plt.title('Segmentace')

    plt.subplot(1, 2, 2)
    plt.imshow(file)
    for center in centers:
        plt.scatter(center[0], center[1], marker='+', color='yellow', s=100)
    plt.title('Těžiště nalezených objektů')

    plt.show()
    plt.close()


def plot_histogram_green_channel(green_img):
    """
    Vykreslení histogramu zelené složky obrázku
    :param green_img: Zelená složka obrázku
    :return: None
    """
    plt.figure(figsize=(8, 6))
    plt.hist(green_img.ravel(), bins=256, color='green', alpha=0.7)
    plt.vlines([100], 0, 255, 'red')
    plt.xlabel('Intenzita zelene slozky')
    plt.ylabel('Počet pixelů')
    plt.title('Histogram zelene slozky obrázku')
    plt.show()


def perform_segmentation(file):
    """
    Provedení segmentace objektů na obrázku pomocí barevné složky g = (G*255)/(R+G+B)
    :param file: Vstupní obrázek
    :return: Segmentovaný obrázek
    """
    file = file.astype('float')
    # g = (G*255) / (R+G+B)
    green = file[:, :, 1] * 255 / (file[:, :, 0] + file[:, :, 1] + file[:, :, 2])
    green = green.astype('uint8')

    segmentation = apply_threshold(green, 100)
    plot_histogram_green_channel(green)
    return segmentation


def analyze_segmentation(seg):
    """
    Závěrečná analýza segmentace
    :param seg: Segmentovaný obrázek
    :return: Barva segmentace, seznam středů objektů
    """
    print("Probíhá identifikace objektů...")
    (color, centers, areas) = segment_objects(seg)

    classifications = classify_coins(color, centers, areas)
    print("Identifikace objektů dokončena.")
    print("Probíhá klasifikace mincí...")
    for (c, center) in zip(classifications, centers):
        if c == 1:
            print(f"Na souřadnici těžiště {center} byla identifikována koruna.")
        else:
            print(f"Na souřadnici těžiště {center} byla identifikována pětikoruna.")
    print(f"Celková hodnota identifikovaných mincí: {sum(classifications)} Kč")

    return color, centers


if __name__ == '__main__':
    """
    Hlavní metoda programu
    """
    file = cv2.imread('cv07_segmentace.bmp')

    segmentation_result = perform_segmentation(file)
    (coloring, centers) = analyze_segmentation(segmentation_result)

    plot_segmentation(segmentation_result, coloring, centers, file)
