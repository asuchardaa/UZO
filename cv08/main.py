import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_channel(image):
    image = image.astype(np.float32)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return g


def apply_threshold(channel, threshold):
    binary_image = np.zeros_like(channel)
    binary_image[channel > threshold] = 0
    binary_image[channel <= threshold] = 255
    return binary_image


def fill(binary_img, output_img, i, j, label):
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        if 0 <= x < binary_img.shape[0] and 0 <= y < binary_img.shape[1]:
            if binary_img[x, y] == 255 and output_img[x, y] == 0:
                output_img[x, y] = label
                stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])


def label_objects(binary_img):
    label = 0
    output_img = np.zeros_like(binary_img)
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i, j] == 255 and output_img[i, j] == 0:
                label += 1
                fill(binary_img, output_img, i, j, label)
    return output_img, label


def morphology(binary_img):
    kernel = np.ones((5, 5), np.uint8)
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    return opened_img


def calculate_centroids(label_img, num_labels):
    centroids = []
    for label in range(1, num_labels + 1):
        ys, xs = np.where(label_img == label)
        centroid = (np.mean(xs), np.mean(ys))
        centroids.append(centroid)
    return centroids


def draw_centroids(image, centroids):
    image = image.copy()
    for centroid in centroids:
        cv2.drawMarker(image, (int(centroid[0]), int(centroid[1])), (0, 255, 0), cv2.MARKER_CROSS, 8, 2)
    return image


def process_image(image_path, threshold):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green_channel = calculate_channel(image)
    binary_image = apply_threshold(green_channel, threshold)
    binary_image = morphology(binary_image)

    labeled_image, num_labels = label_objects(binary_image)
    centroids = calculate_centroids(labeled_image, num_labels)

    return binary_image, image, centroids


def main():
    image_paths = ['cv08_im1.bmp', 'cv08_im2.bmp']
    thresholds = [60, 82]

    fig, axes = plt.subplots(len(image_paths), 3, figsize=(8, 5))

    for i, (image_path, threshold) in enumerate(zip(image_paths, thresholds)):
        binary_image, original_image, centroids = process_image(image_path, threshold)

        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title('Original')

        axes[i, 1].imshow(binary_image, cmap='gray')
        axes[i, 1].set_title('Binary')

        centroid_image = draw_centroids(original_image, centroids)
        axes[i, 2].imshow(centroid_image)
        axes[i, 2].set_title('Centroids')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
