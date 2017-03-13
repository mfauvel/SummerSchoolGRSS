import rasterTools as rt
import scipy as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load data set
X,y=rt.get_samples_from_roi('../Data/university.tif','../Data/university_gt.tif')
wave = sp.loadtxt('../Data/waves.csv',delimiter=',')

# Select the same number of samples
nt = 900
xt,yt=[],[]
for i in sp.unique(y):
    t = sp.where(y==i)[0]
    nc = t.size
    rp =  sp.random.permutation(nc)
    xt.extend(X[t[rp[0:nt]],:])
    yt.extend(y[t[rp[0:nt]]])

xt = sp.asarray(xt)
yt = sp.asarray(yt)
print xt.shape
print yt.shape

# Do LDA
lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
lda.fit(xt,yt.ravel())

# Plot explained variance
l = lda.explained_variance_ratio_
cl= l.cumsum()

for i in range(y.max()-1):
    print "({0},{1})".format(i+1,l[i])

for i in range(y.max()-1):
    print "({0},{1})".format(i+1,cl[i])

# Projet data
import matplotlib.pyplot as plt
xp=lda.transform(xt)

# Save projection
D = sp.concatenate((xp[::10,:4],yt[::10]),axis=1)
sp.savetxt("../FeatureExtraction/figures/lda_proj.csv",D,delimiter=',')

# Save Eigenvectors
D = sp.concatenate((wave[:,sp.newaxis],lda.coef_[:3,:].T),axis=1)
sp.savetxt('../FeatureExtraction/figures/lda_pcs.csv',D,delimiter=',')

im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape
im.shape=(h*w,b)
imp = lda.transform(im)[:,:3]
imp.shape = (h,w,3)

# Save image
rt.write_data('../Data/lda_university.tif',imp,GeoT,Proj)
