import rasterTools as rt
import scipy as sp
from scipy.stats import mode

# Load Thematic Map
im,GeoT,Proj = rt.open_data('../Data/tm_university_svm.tif')
out = sp.empty_like(im)

# Load segmented image
segmented,GeoT,Proj = rt.open_data('../Data/mean_shift_university.tif')

# Do the majority vote
for l in sp.unique(segmented):
    t = sp.where(segmented==l)
    y = im[t]
    out[t] = mode(y, axis=None)[0][0]

# Write the new image
rt.write_data("../Data/tm_university_fusion_mv.tif",out,GeoT,Proj)
