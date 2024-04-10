import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

image = plt.imread('cv04c_robotC.bmp')
image_gray = np.mean(image, axis=2)

laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def conv2d(a, f):
    """
    Vlastní implementovaná funkce pro konvoluci 2D
    """
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def convolve(image, kernel):
    """
    Funkce pro konvoluci obrazu s jádrem
    :param image:
    :param kernel:
    :return:
    """
    return conv2d(image, kernel)


def convolve2d_original(image, kernel):
    """
    Funkce pro konvoluci ale origo import :-)
    :param image:
    :param kernel:
    :return:
    """
    return convolve2d(image, kernel, mode='same', boundary='symm')


def laplacian_edge_detector(image):
    """
    Metoda pro detekci hran Laplaceovým operátorem
    :param image:
    :return:
    """
    return convolve(image, laplacian_kernel)


def sobel_edge_detector(image):
    """
    Funkce pro detekci hran Sobelovým operátorem
    :param image:
    :return:
    """
    gradient_x = convolve2d_original(image, sobel_kernel_x)
    gradient_y = convolve2d_original(image, sobel_kernel_y)
    sobel_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return sobel_magnitude


def kirsch_edge_detector(image):
    """
    Funkce pro detekci hran Kirschovým operátorem
    :param image:
    :return:
    """
    kirsch_filters = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ]
    kirsch_responses = [convolve2d_original(image, kirsch) for kirsch in kirsch_filters]
    kirsch = np.max(kirsch_responses, axis=0)
    return kirsch


def main():
    """
    Hlavní funkce pro zobrazení výsledků
    :return:
    """
    laplace_result = laplacian_edge_detector(image_gray)
    sobel_result = sobel_edge_detector(image_gray)
    kirsch_result = kirsch_edge_detector(image_gray)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Original')
    spectrum = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(image_gray))))
    plt.subplot(1, 2, 2)
    plt.imshow(spectrum, cmap='jet')
    plt.colorbar()
    plt.title('Spectrum')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(laplace_result, cmap='jet')
    plt.colorbar()
    plt.title('Laplace')
    spectrum = np.log(25 + np.abs(np.fft.fftshift(np.fft.fft2(laplace_result))))
    plt.subplot(1, 2, 2)
    plt.imshow(spectrum, cmap='jet')
    plt.colorbar()
    plt.title('Spectrum')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sobel_result, cmap='jet')
    plt.colorbar()
    plt.title('Sobel')
    plt.subplot(1, 2, 2)
    spectrum = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(sobel_result))))
    plt.imshow(spectrum, cmap='jet')
    plt.colorbar()
    plt.title('Spectrum')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(kirsch_result, cmap='jet')
    plt.colorbar()
    plt.title('Kirsch')
    plt.subplot(1, 2, 2)
    spectrum = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(kirsch_result))))
    plt.imshow(spectrum, cmap='jet')
    plt.colorbar()
    plt.title('Spectrum')

    plt.show()
    plt.close()


if __name__ == '__main__':
    """
    Hlavní funkce programu
    :return:
    """
    main()
