#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This reads in images and makes simple measurements of the background light levels

# Created 2021 Oct 21 by E.S.


# In[1]:


from astropy.io import fits
import numpy as np
import glob
import os
import scipy
from scipy import signal
import matplotlib.pyplot as plt


# In[2]:


# directory of images which have already been bias-, dark- and flat-corrected

stem = "/Users/bandari/Documents/postdoc_notre_dame/kriz_data/data_20211018/processed_sci_data_registered"
#stem = "/Users/bandari/Documents/postdoc_notre_dame/kriz_data/data_20211019/calibrated_sci_data_preregistration"

'''
dir_src = stem + "/calibrated_sci_data_preregistration"
dir_write = stem + "/calibrated_sci_data_registered"
file_list_lights_src = sorted(glob.glob(dir_src + "/*fit"))
'''


# In[3]:


file_list_lights_src = sorted(glob.glob(stem + "/*fit"))


# In[4]:


# loop over the images, find median

jd_array = np.nan*np.ones(len(file_list_lights_src))
median_per_pix_array = np.nan*np.ones(len(file_list_lights_src))
amass_array = np.nan*np.ones(len(file_list_lights_src))

for i in range(0,len(file_list_lights_src)):

    frame_this_obj = fits.open(file_list_lights_src[i])
    frame_this = frame_this_obj[0].data
    jd = np.mod(frame_this_obj[0].header["JD"],1)
    median_this = np.median(frame_this)
    amass_this = frame_this_obj[0].header["AIRMASS"]

    jd_array[i] = jd
    median_per_pix_array[i] = median_this # 'pix' here is a pixel-squared
    amass_array[i] = amass_this


# In[8]:


# convert counts per pixel to counts per arcsec^2

PS = 1.56 # asec / pix (where 'pix' is the length of one side of a pixel; note different definition from above)
median_per_asec_sq_array = np.divide(median_per_pix_array,PS**2.)


# In[11]:


# counts conversion factor

# brightest star in 20211018 dataset, TYC 4053-1853-1, which is ~9 mag in R: about 600k counts
del_mag_array = -2.5*np.log10(median_per_asec_sq_array/6e5) # this is 'delta mag between counts per pixel and star in R-band'
abs_mag_array = 9+del_mag_array


# In[13]:


fig, ax1 = plt.subplots()
#plt.gca().invert_yaxis()
ax1.plot(jd_array,abs_mag_array,color="k")
#ax1.plot(jd_array,np.add(amass_array))
ax1.set_xlabel("Modulo JD")
ax1.set_ylabel("Mags in R-band per arcsec^2")
ax1.set_ylim(ax1.get_ylim()[::-1])

ax2 = ax1.twinx()
ax2.plot(jd_array,amass_array,linestyle="--",color="red")
ax2.set_ylabel("Airmass", color="red")

plt.suptitle("2021 Oct 18 dataset")
plt.savefig("junk.pdf")


# In[ ]:
