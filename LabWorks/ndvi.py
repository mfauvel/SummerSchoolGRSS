# Labworks GRSS Summer School

# Initialization
import sys
sys.path.insert(0, '../Codes/')

# Do some usefull imports
import rasterTools as rt
import scipy as sp
import matplotlib.pyplot as plt

# The objective of the labwork is to perform the NDVI computation, take a look to its histogram and treshold it manually

# IR and R bands - Python starts at zero !
IR = 82
R = 55

# Load data set
im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape

# Compute the NDVI
# ndvi = # TO BE COMPLETED
ndvi = (im[:,:,IR].astype(float)-im[:,:,R])/(im[:,:,IR].astype(float)+im[:,:,R])

# Plot the histogram
plt.hist(ndvi.flatten(),bins=40)
plt.show()

# Do a simple threshold
t1,t2=0.2,0.6
ims = sp.zeros_like(ndvi)
ims[:,:]= 2
ims[ndvi>t2] = 3
ims[ndvi<t1] = 1

plt.imshow(ims)
plt.colorbar()
plt.show()
