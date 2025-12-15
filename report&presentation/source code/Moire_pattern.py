# this file implements the function to add Moire pattern to the original image
# meanwhile, it also contains the classical method to remove Moire pattern by the notch butterworth high pass filter

# import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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


# define a function to add Moire pattern by downsampling and then resampling
def add_Moire_pattern_downsample(img_name, sample_rate, gray=True):
    # read the astronomy image
    # add the moire's pattern
    # do the downsampling
    if gray:
        Img_astronomy_1 = cv2.imread("./../images/" + img_name, cv2.IMREAD_GRAYSCALE)
        display_img(Img_astronomy_1, "original image of the astronomy image 1", True)
        M, N = Img_astronomy_1.shape
        Img_astronomy_1_reconstruct = np.zeros((M // sample_rate, N // sample_rate))
        for i in range(M):
            for j in range(N):
                if (i % sample_rate == 0 and j % sample_rate == 0):
                    Img_astronomy_1_reconstruct[i // sample_rate - 1, j // sample_rate - 1] = Img_astronomy_1[i, j]
        display_img(Img_astronomy_1_reconstruct, "reconstructed astronomy image 1 after downsampling", True)
        # do the bilinear interpolation
        Img_astronomy_1_Moire_pattern_noise = cv2.resize(Img_astronomy_1_reconstruct,
                                                         dsize=(N, M), interpolation=cv2.INTER_LINEAR)
        display_img(Img_astronomy_1_Moire_pattern_noise, "astronomy image 1 with Moire pattern", True)
    else:
        Img_astronomy_1 = cv2.imread("./../images/" + img_name)
        Img_astronomy_1 = cv2.cvtColor(Img_astronomy_1, cv2.COLOR_BGR2RGB)
        display_img(Img_astronomy_1, "original image of the astronomy image 1", False)
        M, N, K = Img_astronomy_1.shape
        Img_astronomy_1_reconstruct = np.zeros((M // sample_rate, N // sample_rate, K))
        print(Img_astronomy_1_reconstruct.shape)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    if (i % sample_rate == 0 and j % sample_rate == 0):
                        Img_astronomy_1_reconstruct[i // sample_rate - 1, j // sample_rate - 1, k] = Img_astronomy_1[
                            i, j, k]
        display_img(Img_astronomy_1_reconstruct, "reconstructed astronomy image 1 after downsampling", False)
        # do the bilinear interpolation
        Img_astronomy_1_Moire_pattern_noise = cv2.resize(Img_astronomy_1_reconstruct,
                                                         dsize=(N, M), interpolation=cv2.INTER_LINEAR)
        display_img(Img_astronomy_1_Moire_pattern_noise, "astronomy image 1 with Moire pattern", False)
    return Img_astronomy_1_Moire_pattern_noise


# define a function to get the fourier transform of one image's spatial domain,
# and then shift the center
def get_img_fft_shift(img):
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    return img_fft_shift


# display the discrete fourier transform magnitude of the shifted frequency domain response
def show_DFT_magnitude(img_fft_shift, title):
    plt.imshow(np.abs(img_fft_shift), cmap="gray", norm=LogNorm(vmin=5))
    plt.title(title)
    plt.show()


# create a notchpass filter and multiply it with the DFT of the newspaper car
# implement Butterworth high pass filter
def BHPF(M, N, D0, n, uk, vk):
    H_BHPF = np.zeros((M, N))
    D = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D[u, v] = np.sqrt((u - M / 2 - uk) ** 2 + (
                    v - N / 2 - vk) ** 2)  # euclidean distance from (u,v) to ((uk,vk) related to center)
            H_BHPF[u, v] = 1 / (1 + (D0 / D[u, v]) ** (2 * n))
    return H_BHPF


# implement Butterworth notch reject filter
def notchFilter(img, n, UV_set):  # n is the parameter of the Butterworth highpass filter
    # Q is the number of Butterworth highpass filter pairs
    M, N = img.shape
    # initialize the filter
    H_notch = np.zeros((M, N))
    length_set = len(UV_set)
    for i in range(length_set):
        uk, vk, D0 = UV_set[i]
        if i == 0:
            H_notch = BHPF(M, N, D0, n, uk, vk) * BHPF(M, N, D0, n, -uk, -vk)
        else:
            H_notch = H_notch * BHPF(M, N, D0, n, uk, vk) * BHPF(M, N, D0, n, -uk, -vk)
    return H_notch


# use the inverse fourier transformation to view the image after the addition of the Moire pattern
def img_ifft(img_fft_shift):
    Img_inverse_fft_shift = np.fft.ifftshift(img_fft_shift)
    Img_inverse_fft = np.fft.ifft2(Img_inverse_fft_shift)
    Img_inverse_fft = np.abs(Img_inverse_fft)
    return Img_inverse_fft


# borrow Moire pattern from given image degraded by Moire pattern
# add the borrowed pattern in the frequency domain into the original image to create the Moire pattern
def add_Moire_pattern_borrow(img, Moire_pattern_src, UV_set_newspaper):
    # get the shape of the original image
    M, N = img.shape
    # resize the shape of the source image with Moire pattern to match the shape of original image
    Img_Moire = cv2.resize(Moire_pattern_src, dsize=(N, M), interpolation=cv2.INTER_LINEAR)
    # display the fourier transform magnitude of the Moire pattern source image
    Img_Moire_fft_shift = get_img_fft_shift(Img_Moire)
    # show_DFT_magnitude(Img_Moire_fft_shift,
    #                    "discrete fourier transform magnitude of the Moire pattern source image")
    img_fft_shift = get_img_fft_shift(img)
    n = 4
    # get the notch pass filter
    Img_Moire_notch_pass = 1 - notchFilter(Img_Moire, n, UV_set_newspaper)
    # add the Moire pattern from the source image to the original image
    Img_astronomy_fft_shift_added = img_fft_shift + Img_Moire_notch_pass * Img_Moire_fft_shift
    # show_DFT_magnitude(Img_astronomy_fft_shift_added,
    #                    "discrete fourier transform magnitude after the addition of Moire frequency")
    return Img_astronomy_fft_shift_added

def restore_Moire(img_Moire,UV_set_astronomy):
    # restore the degraded noise
    Img_added_Moire_fft_shift = get_img_fft_shift(img_Moire)
    show_DFT_magnitude(Img_added_Moire_fft_shift,
                       "discrete fourier transform of the astronomy image after the addition of Moire pattern")
    # set the parameter n of the notch filter
    n = 1.5

    # use notch butterworth reject filter
    Img_astronomy_notch_reject = notchFilter(img_Moire, n, UV_set_astronomy)

    # multiply the notch butterworth reject filter with the frequency response of the image with Moire pattern addition
    Img_astronomy_1_Moire_restored_fft_shift = Img_added_Moire_fft_shift * Img_astronomy_notch_reject
    # display the magnitude of the filtered image
    show_DFT_magnitude(Img_astronomy_1_Moire_restored_fft_shift,
                       "the discrete fourier transform magnitude of the filtered image")

    # use inverse fourier transformation to get back the restored astronomy image
    Img_Moire_restored = img_ifft(Img_astronomy_1_Moire_restored_fft_shift)

    return Img_Moire_restored

if __name__ == '__main__':
    # read and display the original image
    img_name = "astronomy_img_1.png"
    Img_astronomy = cv2.imread("./../images/" + img_name, cv2.IMREAD_GRAYSCALE)
    display_img(Img_astronomy, "original image of the astronomy image", True)

    # add Moire pattern to the astronomy image
    Img_Moire_name = "car_newsprint.tif"
    Img_with_moire = cv2.imread("./../images/" + Img_Moire_name, cv2.IMREAD_GRAYSCALE)

    # find the locations of the Moire pattern in the frequency domain
    UV_set_newspaper = [(42, -28, 20), (84, -28, 20), (-42, -28, 20), (-84, -28, 20)]

    # get the image frequency domain response added with the Moire pattern
    Img_added_Moire_fft_shift = add_Moire_pattern_borrow(Img_astronomy, Img_with_moire, UV_set_newspaper)

    # get the inverse fourier transform of the image frequency domain response
    Img_added_Moire = img_ifft(Img_added_Moire_fft_shift)

    # display the image with the added Moire pattern
    display_img(Img_added_Moire,"astronomy image added with the Moire pattern")

    # remove the Moire pattern
    # set the parameters of the u_k, v_k, and D0
    # they denote the location we want to reject, and the extent we want to reject it
    UV_set_astronomy = [(42, -28, 20), (84, -28, 20), (-42, -28, 20), (-84, -28, 20),
                        (42, 80, 10), (84, 60, 10), (42, -80, 10), (84, -60, 10)]

    # get the restored image of the degraded image
    Img_Moire_restored=restore_Moire(Img_added_Moire,UV_set_astronomy)

    # display the restored image
    display_img(Img_Moire_restored,"restored image degraded with the Moire pattern")