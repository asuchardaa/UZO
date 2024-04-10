import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_correction(file, etalon_file, c, index):
    """
    Funkce pro jasovoui korekci s poruchou obrazku
    :param file:
    :param etalon_file:
    :param c:
    :param index:
    :return:
    """
    image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    etalon = cv2.cvtColor(cv2.imread(etalon_file), cv2.COLOR_BGR2RGB)

    np.seterr(divide='ignore', invalid='ignore')

    corrected_image = np.divide(np.dot(c, image), etalon).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, num=index)

    axs[0].imshow(image.astype(np.uint8))
    axs[0].set_title('Před')

    axs[1].imshow(etalon.astype(np.uint8))
    axs[1].set_title('Porucha')

    axs[2].imshow(corrected_image)
    axs[2].set_title('Po')


def histogram_equalization(image_path):
    """
    Funkce pro ekvalizaci histogramu
    :param image_path:
    :return:
    """
    original_image = cv2.imread(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    q = (255 / (height * width) * np.cumsum(hist)).astype(np.uint8)

    equ = q[img]

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Původní obrázek')

    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), 256, [0, 256], color='r', alpha=0.5, label='Původní histogram')
    plt.legend(loc='upper right')
    plt.title('Původní histogram')

    plt.subplot(2, 2, 3)
    plt.imshow(equ, cmap='gray')
    plt.title('Ekvalizovaný obrázek')

    plt.subplot(2, 2, 4)
    plt.hist(equ.ravel(), 256, [0, 256], color='b', alpha=0.5, label='Ekvalizovaný')
    plt.legend(loc='upper right')
    plt.title('Ekvalizovaný histogram')


def plot_amplitude_spectrum(image_path):
    """
    Funkce pro zobrazení amplitudového spektra
    :param image_path:
    :return:
    """
    original_image = cv2.imread(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f_transform = np.fft.fft2(img)
    magnitude_spectrum = np.abs(f_transform)
    f_transform_shifted = np.fft.fftshift(magnitude_spectrum)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(np.log(magnitude_spectrum), cmap='jet')
    plt.title('Amplitudové spektrum')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(f_transform_shifted), cmap='jet')
    plt.title('Amplitudové spektrum')
    plt.colorbar()


def apply_filter(image_path, filter_mask_path, title):
    """
    Funkce pro aplikování filtru na obrázek
    :param image_path:
    :param filter_mask_path:
    :param title:
    :return:
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filter_mask = cv2.imread(filter_mask_path, cv2.IMREAD_GRAYSCALE)
    filter_mask = (filter_mask > 0).astype(np.uint8)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    spectrum_filter = np.fft.fftshift(np.fft.fft2(filter_mask))
    # jenom at mi to lip zobrazuje
    spectrum_filter = np.log(np.abs(spectrum_filter) + 1)

    fshift_filtered = fshift * filter_mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(np.log(np.abs(fshift_filtered) + 1), cmap='jet')
    plt.title('Spektrum po aplikaci filtru ' + filter_mask_path)

    plt.subplot(2, 2, 2)
    plt.imshow(img_back, cmap='gray')
    plt.title('Obrázek po filtraci')


def main():
    """
    Funkce všech funkcí :-)
    :return:
    """
    image_correction('cv04_f01.bmp', 'cv04_e01.bmp', 255, 1)
    image_correction('cv04_f02.bmp', 'cv04_e02.bmp', 255, 2)

    histogram_equalization('cv04_rentgen.bmp')

    plot_amplitude_spectrum('cv04c_robotC.bmp')

    apply_filter('cv04c_robotC.bmp', 'cv04c_filtHP.bmp', 'Horní průchodový filtr')
    apply_filter('cv04c_robotC.bmp', 'cv04c_filtHP1.bmp', 'Horní průchodový filtr 1')
    apply_filter('cv04c_robotC.bmp', 'cv04c_filtDP.bmp', 'Dolní průchodový filtr')
    apply_filter('cv04c_robotC.bmp', 'cv04c_filtDP1.bmp', 'Dolní průchodový filtr 1')

    plt.show()
    plt.close('all')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    Hlavní funkce programu
    """
    main()
