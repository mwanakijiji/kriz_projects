#!/usr/bin/env python
# coding: utf-8

# This finds the stars in images of V1432 Aql

# Created 2021 Aug. 19 by E.S.

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import glob
import os


stem_data = "/Users/bandari/Documents/git.repos/kriz_projects/data/2021-08-04/raw/"

# star images
file_names_stars = sorted(glob.glob(stem_data + "V1432_Aql-*fit"))

# darks
file_names_darks = sorted(glob.glob(stem_data + "d*fit"))

# flats
file_names_flats = sorted(glob.glob(stem_data + "flat*fit"))


# make a master dark
cube_dark = np.zeros((len(file_names_darks),512,512))
for num_dark in range(0,len(file_names_darks)):
    data = fits.open(file_names_darks[0])[0].data
    cube_dark[num_dark,:,:] = data
master_dark = np.mean(cube_dark,axis=0)
# write out
hdu = fits.PrimaryHDU(master_dark)
hdul = fits.HDUList([hdu])
fits_new_file_name = "../data/2021-08-04/calibs_reduced/master_dark.fits"
hdul.writeto(fits_new_file_name, overwrite=False)
print("Wrote "+fits_new_file_name)

# make a master flat
cube_flat = np.zeros((len(file_names_flats),512,512))
for num_flat in range(0,len(file_names_flats)):
    data = fits.open(file_names_flats[0])[0].data
    # normalize
    data = np.divide(data,np.median(data))
    cube_flat[num_flat,:,:] = data
master_flat = np.mean(cube_flat,axis=0)
# write out
hdu = fits.PrimaryHDU(master_flat)
hdul = fits.HDUList([hdu])
fits_new_file_name = "../data/2021-08-04/calibs_reduced/master_flat.fits"
hdul.writeto(fits_new_file_name, overwrite=False)
print("Wrote "+fits_new_file_name)


for num_star_frame in range(0,len(file_names_stars)):

    # read in data
    hdul_read = fits.open(file_names_stars[num_star_frame])
    data_pre_dark_flat = hdul_read[0].data
    hdr = hdul_read[0].header

    # dark-subtract and flat-field
    data_pre_flat = np.subtract(data_pre_dark_flat,master_dark)
    data = np.divide(data_pre_flat,master_flat)
    # write out
    hdu = fits.PrimaryHDU(data, header=hdr)
    hdul = fits.HDUList([hdu])
    fits_new_file_name = "../data/2021-08-04/00_dark_subtracted_flat_fielded/" + os.path.basename(file_names_stars[num_star_frame])
    hdul.writeto(fits_new_file_name, overwrite=False)
    print("Wrote "+fits_new_file_name)

    # get some stats for centroiding
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # find the stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(data - median)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output

    # write out
    text_file_name = "output_other/"+str(os.path.basename(file_names_stars[num_star_frame])).split(".")[-2]+".dat"
    ascii.write(sources, text_file_name, overwrite=False)
    print("Wrote out "+text_file_name)


'''
# for plotting
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
           interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
'''
