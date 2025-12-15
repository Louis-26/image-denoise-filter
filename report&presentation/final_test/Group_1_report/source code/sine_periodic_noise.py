# this file implements the function to add the periodic sine noise
# It also includes the classical algorithm to remove the periodic sine noise

# import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt


# display images with the given title, with option of gray or not
def display_img(img_name, title, gray=True):
    if gray:
        plt.imshow(img_name,
                   vmin=0, vmax=255,
                   cmap="gray")
    else:
        plt.imshow(img_name.astype('uint8'), vmin=0, vmax=255)
    plt.title(title)
    plt.show()


def add_periodic_noise(img, mean, amplitude, frequency):
    M, N = img.shape
    noise = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            noise[i, j] = mean + amplitude * np.sin(2 * np.pi * frequency * j)
    return img + noise


if __name__ == '__main__':
    # read and display the original image
    img_name = "astronomy_img_1.png"
    Img_astronomy = cv2.imread("./../images/" + img_name, cv2.IMREAD_GRAYSCALE)
    display_img(Img_astronomy, "original image of the astronomy image", True)

    # add the sine periodic noise to the original image
    Img_astronomy_sine_periodic = add_periodic_noise(Img_astronomy, 0, 20, 0.1)
    display_img(Img_astronomy_sine_periodic, "image added with the sine periodic noise")
