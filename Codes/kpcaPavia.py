import rasterTools as rt
import scipy as sp
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data set
im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape
im.shape=(h*w,b)
wave = sp.loadtxt('../Data/waves.csv',delimiter=',')

# Scale data
sc = StandardScaler()
im = sc.fit_transform(im)

# Do KPCA
kpca = KernelPCA(kernel='rbf',gamma=1.0/b,n_jobs=-1)
kpca.fit(im[::50,:]) # Use a subset of the total pixel number

# Plot explained variance
l = kpca.lambdas_
cl = l.cumsum()/l.sum()
for i in range(10):
    print "({0},{1})".format(i+1,l[i])

for i in range(10):
    print "({0},{1})".format(i+1,cl[i])

# Save Eigenvectors
idx = sp.arange(kpca.alphas_[0,:].size)+1
D = sp.concatenate((idx[:,sp.newaxis],kpca.alphas_[:3,:].T),axis=1)
sp.savetxt('../FeatureExtraction/figures/kpca_pcs.csv',D,delimiter=',')

imp = kpca.transform(im)[:,:3]
imp.shape = (h,w,3)

# Save image
rt.write_data('../Data/kpca_university.tif',imp,GeoT,Proj)
