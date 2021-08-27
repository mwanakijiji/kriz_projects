# This cross-correlates a series of images to measure the linear displacement between them

# Created 2021 Aug. 19 by E.S.

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import scipy.signal
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

stem = "/Users/bandari/Documents/git.repos/kriz_projects/data/2021-08-04/00_dark_subtracted_flat_fielded/"

file_list_stars = np.sort(glob.glob(stem + "*fit"))

# read in first image as a baseline
hdul = fits.open(file_list_stars[0])
baseline_image = hdul[0].data

# initialize dictionary
keyDict = {"file_name","x_off_pix","y_off_pix","error_pix"}
dict_offsets = dict([(key, []) for key in keyDict])

# loop over pupil images and find their displacements relative to the baseline image
for i in range(0,len(file_list_stars)):

    print("Registering " + str(file_list_stars[i]))

    hdul = fits.open(file_list_stars[i])
    comparison_image = hdul[0].data

    offset_image = comparison_image

    # Code below shamelessly cribbed from
    # https://scikit-image.org/docs/0.13.x/auto_examples/transform/plot_register_translation.html

    # subpixel precision
    shift, error, diffphase = register_translation(baseline_image, offset_image, 100)

    '''
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(baseline_image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(offset_image.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Calculate the upsampled DFT, again to show what the algorithm is doing
    # behind the scenes.  Constants correspond to calculated values in routine.
    # See source code for details.
    '''
    image_product = np.fft.fft2(baseline_image) * np.fft.fft2(offset_image).conj()
    cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
    '''
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Supersampled XC sub-area")
    plt.suptitle(os.path.basename(file_list_stars[i])+"\nDetected subpixel offset (y, x): {}".format(shift))

    # save plot
    plt.savefig("output_images/pupil_shift_"+str(int(i))+".png", dpi=400)
    '''
    # update the dictionary
    dict_offsets["file_name"].append(os.path.basename(file_list_stars[i]))
    dict_offsets["x_off_pix"].append(shift[1])
    dict_offsets["y_off_pix"].append(shift[0])
    dict_offsets["error_pix"].append(error)

# convert to DataFrame
df_offsets = pd.DataFrame.from_dict(dict_offsets)

# write out displacements
df_offsets.to_csv("junk_displacements.csv")

# make plot of all displacements
fig = plt.figure(figsize=(10,5))
plt.plot(df_offsets["x_off_pix"], marker="o", label="x offset")
plt.plot(df_offsets["y_off_pix"], marker="o", label="y offset")
plt.plot(df_offsets["error_pix"], marker="o", label="error")
plt.plot(np.sqrt(np.power(df_offsets["x_off_pix"],2)+np.power(df_offsets["y_off_pix"],2)), marker="o", label="total offset")
plt.xlabel("Trial number")
plt.ylabel("Shift (pixels)")
plt.legend()
plt.savefig("output_images/junk_plot.png", dpi=400)
