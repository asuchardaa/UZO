# Autoři: Adam Sucharda, Marcel Horváth @ 2024
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


def read_image(img_name):
    """
    Načte obrázek a převede ho na šedotónový formát.

    :param img_name: Název souboru obrázku.
    :return: Obrázek v RGB a šedotónovém formátu.
    """
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, img_gray


def apply_threshold(img_gray):
    """
    Aplikuje prahování na šedotónový obrázek.

    :param img_gray: Šedotónový obrázek.
    :return: Prahový obrázek.
    """
    _, thresh = cv2.threshold(img_gray, 130, 255, 0)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def morphological_operations(thresh):
    """
    Provádí morfologické operace na prahovém obrázku.

    :param thresh: Prahový obrázek.
    :return: Zavřený obrázek a dilataci.
    """
    kernel = np.ones((6, 6), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    return closing, sure_bg


def calculate_markers(closing, sure_bg):
    """
    Vypočítá markery pro aplikaci watershed algoritmu.

    :param closing: Zavřený obrázek.
    :param sure_bg: Dilatace.
    :return: Markery pro aplikaci watershed a obrázek s jistým popředím.
    """
    # pocitam transformaci vzdalenosti -> nastesti funkce dela vsechno za me
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    # postupuju dle prahovani cele trasnformace vzdalenosti pro ziskani popredi
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    # pozadi (bg) je vse co je za popredim (fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # tady jen identifikuju vsechny nalezene objekty na popredi (fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    return markers, sure_fg


def watershed_segmentation(img, markers):
    """
    Aplikuje algoritmus watershed na obrázek.

    :param img: Obrázek.
    :param markers: Markery pro aplikaci watershed.
    :return: Výsledek aplikace algoritmu watershed.
    """
    markers_result = cv2.watershed(img, markers)
    return markers_result


def calculate_centroids(img, sure_fg):
    """
    Vypočítá težiště na obrázku a značí je.

    :param img: Obrázek.
    :param sure_fg: Obrázek s jistým popředím.
    """
    label_array, num_features = ndimage.label(sure_fg)
    centroid_list = ndimage.center_of_mass(sure_fg, label_array, range(1, num_features + 1))
    num_coins = num_features

    for idx, centroid in enumerate(centroid_list):
        centroid_coordinates = (int(centroid[1]), int(centroid[0]))
        print(f"Mince {idx+1}: Souřadnice težiště: {centroid_coordinates}")
        cv2.drawMarker(img, (int(centroid[1]), int(centroid[0])), (255, 0, 0), markerType=cv2.MARKER_CROSS,
                       markerSize=10, thickness=2)
        cv2.putText(img, str(idx + 1), (int(centroid[1]) + 20, int(centroid[0]) + 20), cv2.QT_FONT_NORMAL, 0.5,
                    (255, 255, 0), 1)

    for i in range(1, num_coins + 1):
        if i not in range(1, num_coins + 1):
            print(f"Mince {i}: Číslo mimo nalezené objekty")


def main():
    """
    Hlavní funkce programu.
    """
    img_name = "cv10_mince.jpg"
    img, img_gray = read_image(img_name)
    thresh = apply_threshold(img_gray)
    closing, sure_bg = morphological_operations(thresh)
    markers, sure_fg = calculate_markers(closing, sure_bg)
    markers_result = watershed_segmentation(img, markers)
    calculate_centroids(img, sure_fg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title('Obrázek s těžišti')
    axes[0].axis('off')

    axes[1].imshow(markers_result, cmap='jet')
    axes[1].set_title('Watershed')
    axes[1].axis('off')

    plt.show()


if __name__ == "__main__":
    """
    Spouštěcí funkce, však ty už víš ;-)
    """
    main()
