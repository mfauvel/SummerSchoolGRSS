import rasterTools as rt
import scipy as sp
import pysptools.eea as eea

# Load data set
im,GeoT,Proj = rt.open_data('../Data/Moffett_full.tif')
[h,w,b]=im.shape
wave = sp.loadtxt('../Data/wave_moffett.csv',delimiter=',')

# NFINDR
nfindr = eea.NFINDR()
Unf = nfindr.extract(im.astype(float), 3, normalize=True)

# Plot endmember
T =  sp.concatenate((wave[:,sp.newaxis],Unf.T),axis=1)
sp.savetxt("../Unmixing/figures/endmembers.csv",T,delimiter=",")
