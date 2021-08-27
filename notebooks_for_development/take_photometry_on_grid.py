#!/usr/bin/env python
# coding: utf-8

# This takes photometry of stars through apertures arranged on a grid which moves
# to compensate for poor telescope guiding

# Created 2021 Aug. 20 by E.S.

import pandas as pd
import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import glob
import os
from numpy.linalg import inv
import svdt

stem_data = "/Users/bandari/Documents/git.repos/kriz_projects/data/2021-08-04/00_dark_subtracted_flat_fielded/"
stem_star_positions = "/Users/bandari/Documents/git.repos/kriz_projects/notebooks_for_development/output_other/"

# names of files
file_names_stars = sorted(glob.glob(stem_data + "V1432_Aql-*fit"))
file_names_pos = sorted(glob.glob(stem_star_positions + "V1432_Aql-*dat"))

# file of measured displacements between images
file_name_disp = "displacements_v1432_aql.csv"


def coords_brightest(star_positions_scrambled_ids_pass, num_rank):
    '''
    Return the coordinates of the brightest stars in the dataframe (column "Flux")

    INPUTS:
    star_positions_scrambled_ids_pass: DataFrame with xcentroid, ycentroid, and flux info
    num_rank: Nth brightest stars returned (i.e., 3 -> coordinates of 3 brightest are returned)

    OUTPUTS:
    2D array of x,y coordinates of the Nth brightest stars
    '''

    brightest_fluxes = np.sort(star_positions_scrambled_ids_pass["flux"])[-int(num_rank):]
    #print("brightest")
    #print(brightest_fluxes)
    # pick table rows where stars are at least as bright as the third-brightest star
    idx_cond = star_positions_scrambled_ids_pass["flux"]>=brightest_fluxes[0]
    # get their positions
    x_brights = star_positions_scrambled_ids_pass["xcentroid"][idx_cond]
    y_brights = star_positions_scrambled_ids_pass["ycentroid"][idx_cond]
    #print(star_positions_scrambled_ids_pass[idx_cond])
    coords_empir_brights = zip(x_brights,y_brights)

    return list(coords_empir_brights)


def match_stars(predicted_coords_pass, empirical_coords_pass):
    '''
    Matches up predicted and empirical coordinates of bright stars

    INPUTS:
    predicted_coords_pass: list of predicted positions of the brightest stars in the initial frame
    empirical_coords_pass: list of empirical coordinates of LOTS of stars in the current frame

    OUTPUTS:
    df_map: DataFrame with mapping of predicted and true coordinates
    '''

    # convert inputs to dataframes
    df_pred = pd.DataFrame(predicted_coords_pass, columns=["x_pred", "y_pred"])
    df_empir = pd.DataFrame(empirical_coords_pass, columns=["x_empir", "y_empir"])

    # initialize columns of 'true' coordinates to predicted dataframe
    df_pred["x_true"] = np.nan
    df_pred["y_true"] = np.nan


    # pluck out the 'true' coordinates from the empirical data
    for num_bright in range(0,len(df_pred)):
        # condition is "where difference in coordinates is less than N=3 pixels"
        idx_cond = np.sqrt(
                        np.add(
                            np.power(np.subtract(df_pred["x_pred"][num_bright],df_empir["x_empir"]),2.),
                            np.power(np.subtract(df_pred["y_pred"][num_bright],df_empir["y_empir"]),2.)
                            )
                        ) < 3

        # populate the 'true' columns of the DataFrame of predicted values
        df_pred["x_true"][num_bright] = df_empir["x_empir"].loc[idx_cond].values[0]
        df_pred["y_true"][num_bright] = df_empir["y_empir"].loc[idx_cond].values[0]

    df_map = df_pred.copy(deep=True)

    return df_map


def transform_coords(T_trans, R_rot, array_pre_transform):
    '''
    Applies translation and rotation to input array.

    INPUTS:
    array_pre_transform: 2D array before transformation [numpy ndarray (Nx3)]
    T_trans: translation array
    R_rot: rotation array

    OUTPUT:
    array_post_transform: the input 2D array, after translation and rotation
    '''

    # rotate first
    array_post_rotate = np.dot(R,array_pre_transform.T).T # note the transposes here
    # then translate
    array_post_translate = np.stack([np.add(coord_set,T_trans) for coord_set in array_post_rotate])

    return array_post_translate


# read in image displacements for ALL stars in ALL the frames
disp_images = pd.read_csv(stem_star_positions+file_name_disp)
# read in first image's star locations (but with scrambled IDs)
initial_star_pos = pd.read_csv(file_names_pos[0], delim_whitespace=True)

# find the coordinates of the brightest three stars in the initial image;
# we will use this to figure out the net translation and rotation of successive frames
initial_coords_empir_brights = coords_brightest(initial_star_pos,num_rank=3)


# machinery to test
'''
#test_predicted_xyz = np.array(([np.sqrt(2)/2.,-np.sqrt(2)/2.,0],[1,0,0],[np.sqrt(2)/2.,np.sqrt(2)/2.,0],[0,1,0]))
#test_true_xyz = np.array(([1,0,0],[np.sqrt(2)/2.,np.sqrt(2)/2.,0],[0,1,0],[-np.sqrt(2)/2.,np.sqrt(2)/2.,0]))
#test_true_xyz = np.array(([0,-1,0],[-1,0,0],[0,1,0],[0,1,0])) # rotation, no translation
#test_true_xyz = np.array(([-2,1,0],[-1,2,0],[0,1,0],[0,1,0])) # translation, no rotation
#test_true_xyz = np.array(([0,-2,0],[-1,-1,0],[0,0,0],[0,0,0])) # rotation and translation
print("predicted:")
print(test_predicted_xyz)
print("true:")
print(test_true_xyz)

# generate the transformations from the data
R, T, RMSE = svdt.svd(A=test_predicted_xyz, B=test_true_xyz)

# B = R*A + L + err.

# test the function
transformed_grid_xyz = transform_coords(T_trans=T, R_rot=R, array_pre_transform=test_predicted_xyz)

print("transformed")
print(transformed_grid_xyz.round(decimals=2))
'''


for num_star_file in range(0,len(file_names_stars)):

    # for each frame, calculate needed shifts of the aperture grid and take photometry
    print("Doing file number " + str(num_star_file))

    # make copy of DataFrame of initial stars (we will change the x,y coords to make a prediction)
    predicted_star_pos = initial_star_pos.copy(deep=True)

    # read in current star image
    star_image = fits.open(file_names_stars[num_star_file])[0].data

    # read in empirically-found positions and photometry of ALL stars in this frame (but with scrambled IDs)
    star_info_scrambled_ids = pd.read_csv(file_names_pos[num_star_file], delim_whitespace=True)

    # retrieve (x,y) linear displacement vectors found independently for ALL stars in that frame, as was
    # found by cross-correlating images
    idx = disp_images["file_name"]==str(os.path.basename(file_names_stars[num_star_file]))
    x_disp = disp_images["x_off_pix"].where(idx).dropna().values[0]
    y_disp = disp_images["y_off_pix"].where(idx).dropna().values[0]

    # shift aperture grid of first (i.e., [0]) array of images by the linear displacement; this is a first-order
    # correction to fit the current frame (n.b. we use this grid and translate it in order to preserve the ID
    # information of the stars; otherwise, if we re-find the stars in each frame we don't know which is which)
    predicted_star_pos["xcentroid"] = initial_star_pos["xcentroid"]-x_disp
    predicted_star_pos["ycentroid"] = initial_star_pos["ycentroid"]-y_disp
    predicted_coords = list(zip(predicted_star_pos["xcentroid"],predicted_star_pos["ycentroid"]))

    # find the predicted coordinates (in this image) of the brightest three stars (as measured in the initial image)
    predicted_coords_brights = coords_brightest(predicted_star_pos,num_rank=3)

    # find the empirical coordinates of the brightest *twenty* stars in this image
    # (here we choose twenty, and from those we'll match the correct three, because the brightest
    # three stars in the first image are not necessarily the brightest three in every image)
    empirical_coords_brights = coords_brightest(star_info_scrambled_ids,num_rank=30)

    # match the stars by finding the sets of predicted and empirical coordinates that are within 3 (or so) pixels
    coord_mapping_brightest = match_stars(predicted_coords_brights, empirical_coords_brights)

    # use SVD decomposition to find the translation and rotation to minimize the residuals
    # Note syntax here:
        # Matrix of estimated, translated positions    A: [ [x_1 y_1 z_1=0], [x_2 y_2 z_2=0], [x_3 y_3 z_3=0] ]
        # Matrix of empirically found positions        B: [ [x_1' y_1' z_1'=0], [x_2' y_2' z_2'=0], [x_3' y_3' z_3'=0] ]
        # Rotation matrix                              R
        # Translation matrix                           T
    predicted_array = np.array([[coord_mapping_brightest["x_pred"][0], coord_mapping_brightest["y_pred"][0], 0.],
                           [coord_mapping_brightest["x_pred"][1], coord_mapping_brightest["y_pred"][1], 0.],
                           [coord_mapping_brightest["x_pred"][2], coord_mapping_brightest["y_pred"][2], 0.]])
    true_array = np.array([[coord_mapping_brightest["x_true"][0], coord_mapping_brightest["y_true"][0], 0.],
                           [coord_mapping_brightest["x_true"][1], coord_mapping_brightest["y_true"][1], 0.],
                           [coord_mapping_brightest["x_true"][2], coord_mapping_brightest["y_true"][2], 0.]])
    R, T, RMSE = svdt.svd(A=predicted_array, B=true_array)

    # transform the full aperture grid with predicted positions
    # translation, then rotation
    predicted_grid_xy = np.array(predicted_star_pos[["xcentroid" ,"ycentroid"]])
    # append z=0
    predicted_grid_xyz = np.append(predicted_grid_xy,np.zeros((len(predicted_star_pos["xcentroid"]),1)),1)

    # rotate and translate
    transformed_grid_xyz = transform_coords(T,R,predicted_grid_xyz)

    # for photometry, we just want the x,y coordinates (not z)
    predicted_apertures = [[coords[0],coords[1]] for coords in predicted_grid_xyz]
    post_transform_apertures = [[coords[0],coords[1]] for coords in transformed_grid_xyz]

    # do the photometry
    '''
    CHOICE OF APERTURES: THE PREDICTED ONES (TRANSLATION COMPENSATION ONLY), OR
    PREDICTION WITH TRANSLATION AND ROTATION TRANSFORMATION CORRECTION; COMMENT
    OUT THE LINES BELOW AS NEEDED
    '''
    aperture = CircularAperture(positions=predicted_apertures, r=3.)
    annulus_aperture = CircularAnnulus(positions=predicted_apertures, r_in=6., r_out=8.) # for determining sky background
    #aperture = CircularAperture(positions=post_transform_apertures, r=3.)
    #annulus_aperture = CircularAnnulus(positions=post_transform_apertures, r_in=6., r_out=8.) # for determining sky background
    apers = [aperture, annulus_aperture]
    phot_table = aperture_photometry(star_image, apers)

    # the photometry table will have
    #   aperture_sum_0: the sum of counts in the central aperture
    #   aperture_sum_1: the sum of counts inside the annulus
    # to subtract the sky background, find the average counts in pixels within the
    # annulus, then calculate the background within the central annulus, then
    # subtract that from aperture_sum_0
    # (this snipped of code from https://photutils.readthedocs.io/en/stable/aperture.html)
    bkg_mean = phot_table["aperture_sum_1"] / annulus_aperture.area
    bkg_sum = bkg_mean * aperture.area
    final_sum = phot_table["aperture_sum_0"] - bkg_sum
    phot_table["residual_aperture_sum"] = final_sum
    phot_table["residual_aperture_sum"].info.format = '%.8g'  # for consistent table output


    # convert to DataFrame
    df_phot = phot_table.to_pandas()

    # write photometry to text file
    text_file_name = "output_other/aper_photom_"+str(os.path.basename(file_names_stars[num_star_file])).split(".")[-2]+".dat"
    ascii.write(phot_table, text_file_name, overwrite=True)
    print("Wrote out "+text_file_name)

    ## show FYI plot
    '''
    apertures_pred = CircularAperture(predicted_apertures, r=4.)
    apertures_post_transf = CircularAperture(post_transform_apertures, r=4.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(star_image, cmap='Greys_r', origin='lower', norm=norm,
               interpolation='nearest')
    apertures_pred.plot(color='red', lw=1.5, alpha=0.5)
    apertures_post_transf.plot(color='blue', lw=1.5, alpha=0.5)
    plt.show()
    '''

    # write FYI plot
    '''
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(star_image, cmap='Greys', origin='lower', norm=norm,
               interpolation='nearest')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    '''
