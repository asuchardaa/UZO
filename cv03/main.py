# Autor: Adam Sucharda, 2024
import cv2
import numpy as np
import math


def calculate_dimensions(image, angle):
    """
    Metoda pro kalkulaci dimenze rotovanyho obrazu
    :param image:
    :param angle:
    :return:
    """
    radians = math.radians(angle)
    height_rotated = round(abs(image.shape[0] * math.cos(radians))) + round(abs(image.shape[1] * math.sin(radians)))
    width_rotated = round(abs(image.shape[1] * math.cos(radians))) + round(abs(image.shape[0] * math.sin(radians)))
    return height_rotated, width_rotated


def create_rotated_image(height, width, channels):
    """
    Funkce pro prazdnotu (obrazek)
    :param height:
    :param width:
    :param channels:
    :return:
    """
    return np.uint8(np.zeros((height, width, channels)))


def calculate_centers(image):
    """
    Metoda pro vypocet strednich hodnot vstupniho obrazku
    :param image:
    :return:
    """
    center_x, center_y = (image.shape[1] // 2, image.shape[0] // 2)
    return center_x, center_y


def calculate_rotated_centers(width_rotated, height_rotated):
    """
    Funkce pro stredni hodnotu otocenyho obrazku
    :param width_rotated:
    :param height_rotated:
    :return:
    """
    center_rotated_x, center_rotated_y = (width_rotated // 2, height_rotated // 2)
    return center_rotated_x, center_rotated_y


def rotate_pixel(image, i, j, center_rotated_x, center_rotated_y, radians, center_y, center_x):
    """
    Metoda pro rotaci pixelu vstupniho obrazku, chci najit jeho pozici v otocenem obrazku
    :param image:
    :param i:
    :param j:
    :param center_rotated_x:
    :param center_rotated_y:
    :param radians:
    :param center_y:
    :param center_x:
    :return:
    """
    x = (i - center_rotated_x) * math.cos(radians) + (j - center_rotated_y) * math.sin(radians)
    y = -(i - center_rotated_x) * math.sin(radians) + (j - center_rotated_y) * math.cos(radians)
    x = round(x) + center_y
    y = round(y) + center_x
    return x, y


def rotate_image(image, angle):
    """
    Metoda pro rotaci obrazku
    :param image:
    :param angle:
    :return:
    """
    height_rotated, width_rotated = calculate_dimensions(image, angle)
    rotated_image = create_rotated_image(height_rotated, width_rotated, image.shape[2])
    center_x, center_y = calculate_centers(image)
    center_rotated_x, center_rotated_y = calculate_rotated_centers(width_rotated, height_rotated)

    for i in range(rotated_image.shape[0]):
        for j in range(rotated_image.shape[1]):
            x, y = rotate_pixel(image, i, j, center_rotated_x, center_rotated_y, math.radians(angle), center_y, center_x)

            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                rotated_image[i, j, :] = image[x, y, :]

    return rotated_image


def display_image(image, window_name):
    """
    Funkce pro vykresleni obrazku
    :param image:
    :param window_name:
    :return:
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


if __name__ == "__main__":
    """
    Hlavni funkce programu
    """
    img = cv2.imread('cv03_robot.bmp')

    display_image(img, "Original")

    # zamerne testuju po 30 stupnich, vysledek neni marginalni
    # a prece jen je tam videt, jestli to je dobhre delany nebo ne.
    # Navic obsahuje uhly pro kontrolu (30, 90, 180, 270, 360)
    # for angle in range(30, 361, 30):
    #     rotated_image = rotate_image(img, angle)
    #     display_image(rotated_image, f"Otoceno o {angle} stupnu")

    angle = 330
    rotated_image = rotate_image(img, angle)
    display_image(rotated_image, f"Otoceno o {angle} stupnu")

    cv2.destroyAllWindows()
