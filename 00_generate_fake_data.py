#!/usr/bin/env python
# coding: utf-8

# This is for generating fake data for testing a photometry pipeline

# Created 2021 Aug 6 by E.S.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pytransit import QuadraticModel


# simple Gaussian white noise
length_dataset = 2000

# time; don't know what form this will take
abcissa_time = np.linspace(0,1.2,length_dataset)
# Gaussian noise
photometry_white_zero = np.random.normal(loc=0.0, scale=0.01, size=length_dataset)
photometry_norm_white = np.add(1.,photometry_white_zero)

plt.plot(abcissa_time,photometry_norm_white)
plt.ylim([0,1.2])
plt.title("White noise")
plt.xlabel("Time")
plt.ylabel("Normalized photometry")
plt.show()

#pd.to_csv("junk_simple_gaussian.csv")


# generate transit model

tm = QuadraticModel()
#tm.set_data(times)
tm.set_data(abcissa_time)

# note t0 is time-of-center
model_transit = tm.evaluate(k=0.5, ldc=[0.2, 0.1], t0=0.6, p=2.7, a=3.0, i=0.5*np.pi)

noisy_transit = np.add(photometry_white_zero,model_transit)

plt.clf()
plt.plot(tm.time, noisy_transit)
plt.ylim([0,1.2])
plt.title("Transit + white noise")
plt.xlabel("Time")
plt.ylabel("Fake transit")
plt.show()
