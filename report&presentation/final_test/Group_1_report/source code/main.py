# this file is the main file to execute the whole project,
# we can see the overall process of these images' degradation and restoration

# import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, filters
from skimage.morphology import disk
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import Moire_pattern
import sine_periodic_noise
import gaussian_noise


# display images with the given title, with option of gray or not
def display_img(img_name, title, gray=True, v_range=True):
    # v_range denotes whether the range of intensity values is 0-255
    if v_range:
        if gray:
            plt.imshow(img_name, vmin=0, vmax=255, cmap="gray")
        else:
            plt.imshow(img_name.astype('uint8'), vmin=0, vmax=255)
        plt.title(title)
        plt.show()
    else:
        if gray:
            plt.imshow(img_name, cmap="gray")
        else:
            plt.imshow(img_name.astype('uint8'), vmin=0, vmax=255)
        plt.title(title)
        plt.show()


# display the result, and evaluate the result by PSNR and SSIM
def evaluate_result(img_true, img_test, name, v_range=True):
    display_img(img_test, name, True, v_range=v_range)
    print(f"The PSNR of the {name}: {psnr(img_true, img_test, data_range=255)}")
    print(f"The SSIM of the {name}: {ssim(img_true, img_test, data_range=255)}")
    print()


# define notch filter
def notch_filter(Img_noisy, UV_set_astronomy):
    # find the fourier transform magnitude of the noisy image
    Img_noisy_fft_shift = Moire_pattern.get_img_fft_shift(Img_noisy)
    # Moire_pattern.show_DFT_magnitude(Img_noisy_fft_shift,
    #                                  "discrete fourier transform magnitude of the noisy image")

    # set the parameters of the u_k, v_k, and D0
    # they denote the location we want to reject, and the extent we want to reject it

    # set the parameter n of the notch filter
    n = 1.5

    # use notch butterworth reject filter
    Img_astronomy_notch_reject = Moire_pattern.notchFilter(Img_noisy, n, UV_set_astronomy)

    # get the fourier transform of the noisy image
    # Img_noisy_fft_shift = Moire_pattern.get_img_fft_shift(Img_noisy)

    # multiply the notch butterworth reject filter with the frequency response of the image with Moire pattern addition
    Img_restore_notch_fft_shift = Img_noisy_fft_shift * Img_astronomy_notch_reject

    # find out the modified fourier transformation domain
    # Moire_pattern.show_DFT_magnitude(Img_restore_notch_fft_shift,
    #                                  "discrete fourier transform magnitude of the restored image")

    # use inverse fourier transformation to get back the restored astronomy image
    Img_restore_notch = Moire_pattern.img_ifft(Img_noisy_fft_shift)
    return Img_restore_notch


def restore_img(img_name):
    # read and display the original image
    # img_name = "astronomy_img_1.png"
    Img_astronomy = cv2.imread("./../images/" + img_name, cv2.IMREAD_GRAYSCALE)
    display_img(Img_astronomy, "original astronomy image", True)

    # add the different noises to the image and generate the noisy image
    # add Moire pattern to the original image
    Img_Moire_name = "car_newsprint.tif"
    Img_with_moire = cv2.imread("./../images/" + Img_Moire_name, cv2.IMREAD_GRAYSCALE)

    # find the locations of the Moire pattern in the frequency domain
    UV_set_newspaper = [(42, -28, 20), (84, -28, 20), (-42, -28, 20), (-84, -28, 20)]

    # get the image frequency domain response added with the Moire pattern
    Img_added_Moire_fft_shift = Moire_pattern.add_Moire_pattern_borrow(Img_astronomy, Img_with_moire, UV_set_newspaper)

    # get the inverse fourier transform of the image frequency domain response
    Img_added_Moire = Moire_pattern.img_ifft(Img_added_Moire_fft_shift)

    # display the image with the added Moire pattern
    display_img(Img_added_Moire, "astronomy image added with the Moire pattern")

    # add the sine periodic noise to the astronomy image
    Img_astronomy_sine_periodic = sine_periodic_noise.add_periodic_noise(Img_added_Moire, 0, 20, 0.1)
    display_img(Img_astronomy_sine_periodic, "image added with the sine periodic noise")

    # add the gaussian noise to the astronomy image
    Img_noisy = gaussian_noise.add_gaussian_noise(Img_astronomy_sine_periodic, 0, 15)
    # display_img(Img_noisy, "the noisy image degraded with three kinds of noise")

    # compute the PSNR and SSIM of the noisy image
    evaluate_result(Img_astronomy, Img_noisy, "noisy image")

    # use classical algorithms to try to restore the noisy image
    # in spatial domain

    # use average filter
    Img_restore_mean = cv2.blur(Img_noisy, (3, 3))
    evaluate_result(Img_astronomy, Img_restore_mean, "image restored by mean filter")

    # use median filter
    Img_restore_median = filters.median(Img_noisy, disk(3))
    evaluate_result(Img_astronomy, Img_restore_median, "image restored by median filter")

    # use adaptive median filter
    Img_restore_adaptive_median = filters.threshold_local(Img_noisy, 3, "median")
    evaluate_result(Img_astronomy, Img_restore_adaptive_median,
                    "image restored by adaptive median filter")

    # use gaussian filter
    Img_restore_gaussian = filters.gaussian(Img_noisy, 3)
    evaluate_result(Img_astronomy, Img_restore_gaussian,
                    "image restored by gaussian filter")

    # in frequency domain
    PSF = np.ones((5, 5)) / 25
    # use wiener filter
    Img_restore_wiener = restoration.wiener(Img_noisy / 255, PSF, 1.09)
    evaluate_result(Img_astronomy, Img_restore_wiener * 255,
                    "image restored by wiener filter", False)
    # print("wiener",Img_restore_wiener)

    # use richardson-lucy algorithm filter
    Img_restore_RL = restoration.richardson_lucy(Img_noisy / 255, PSF, num_iter=5)
    evaluate_result(Img_astronomy, Img_restore_RL * 255,
                    "image restored by richardson-lucy algorithm filter", False)

    # print("RL",Img_restore_RL)

    # use notch filter

    UV_set_astronomy = [(42, -28, 20), (84, -28, 20), (-42, -28, 20), (-84, -28, 20)]
    Img_restore_notch = notch_filter(Img_noisy, UV_set_astronomy)
    evaluate_result(Img_astronomy, Img_restore_notch, "image restored by notch filter")

    # innovative algorithm: use multiple filters with a proper combination,

    # laplacian filter sharpening
    # gaussian filter result added with laplacian filter
    k = 3
    Img_restore_gaussian_laplacian_boost = filters.laplace(Img_restore_gaussian)
    Img_restore_gaussian_laplacian = Img_restore_gaussian + k * Img_restore_gaussian_laplacian_boost
    evaluate_result(Img_astronomy, Img_restore_gaussian_laplacian,
                    "image restored by gaussian and laplacian filter")

    # notch filter result added with laplacian filter
    k = 1
    Img_restore_notch_laplacian_boost = filters.laplace(Img_restore_notch)
    Img_restore_notch_laplacian = Img_restore_notch + k * Img_restore_notch_laplacian_boost
    evaluate_result(Img_astronomy, Img_restore_notch_laplacian,
                    "image restored by notch and laplacian filter")

    # wiener filter result added with laplacian filter
    k = 2
    Img_restore_wiener_255 = Img_restore_wiener * 255
    Img_restore_wiener_laplacian_boost = filters.laplace(Img_restore_wiener_255)
    Img_restore_wiener_laplacian = Img_restore_wiener_255 + k * Img_restore_wiener_laplacian_boost
    evaluate_result(Img_astronomy, Img_restore_wiener_laplacian,
                    "image restored by wiener and laplacian filter")

    # Histogram equalization
    # use histogram equalization on restored results of gaussian filter
    Img_restore_gaussian_hist = cv2.equalizeHist(np.uint8(Img_restore_gaussian))
    evaluate_result(Img_astronomy, Img_restore_gaussian_hist,
                    "image restored by gaussian filter and histogram equalization")

    # use histogram equalization on restored results of notch filter
    Img_restore_notch_hist = cv2.equalizeHist(np.uint8(Img_restore_notch))
    evaluate_result(Img_astronomy, Img_restore_notch_hist,
                    "image restored by notch filter and histogram equalization")

    # use histogram equalization on restored results of wiener filter
    Img_restore_wiener_hist = cv2.equalizeHist(np.uint8(Img_restore_wiener_255))
    evaluate_result(Img_astronomy, Img_restore_wiener_hist,
                    "image restored by wiener filter and histogram equalization")

    # wiener filter combined with notch filter
    Img_restore_wiener_notch = notch_filter(Img_restore_wiener_255, UV_set_astronomy)
    evaluate_result(Img_astronomy, Img_restore_wiener_notch,
                    "image restored by wiener filter and notch filter")

    # gaussian filter combined with notch filter
    Img_restore_gaussian_notch = notch_filter(Img_restore_gaussian, UV_set_astronomy)
    evaluate_result(Img_astronomy, Img_restore_gaussian_notch,
                    "image restored by gaussian filter and notch filter")


# restore_img("astronomy_img_1.png")
# restore_img("astronomy_img_2.jpg")
restore_img("astronomy_img_3.jpg")

