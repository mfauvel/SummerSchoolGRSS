# Labworks GRSS Summer School

# Initialization
import sys
sys.path.insert(0, '../Codes/')

# Do some usefull imports
import rasterTools as rt
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg

def skip_extrem(im):
    # Get percentile
    pm,pM = sp.percentile(im, [2,98])
    ims = im
    ims[im<pm] = pm
    ims[im>pM] = pM
    return ims

# The objective of the first labwork is to get familiar with the data
# You will vizualise two bands, compute the median value of differences, compute the covariance matrix of the images and its condition number.

# Choose two bands for vizualisation in [0, 102]
b1= 40
b2= 41
    
# Load data set
im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape

# Visualization of band 1
plt.figure()
ims = skip_extrem(im[:,:,b1])
plt.imshow(ims,cmap="gray")
plt.colorbar()

# Visualization of band 2
plt.figure()
ims = skip_extrem(im[:,:,b2])
plt.imshow(ims,cmap="gray")
plt.colorbar()

# Mean differences
print "Median value of differences between band {} and {} is {}".format(b1,b2,100.*sp.median((im[:,:,b2].astype(float)-im[:,:,b1])/im[:,:,b1]))

# Computation of the correlation
im.shape = (h*w,b)
cov = sp.cov(im[::4,:],bias=1,rowvar=0)
dcov = sp.sqrt(sp.diag(cov))
cor = cov/dcov[:,sp.newaxis]
cor /= dcov[sp.newaxis,:]
plt.figure()
plt.imshow(cor,interpolation='nearest')
plt.colorbar()

# Compute condition number
s = linalg.svd(cov, compute_uv=False)
print("Condition number is {}".format(s[0]/s[-1]))

# Plot all the figures
plt.show()

