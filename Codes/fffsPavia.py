import rasterTools as rt
import scipy as sp
import npfs as npfs

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

# Do FFFS
maxVar = 12
model = npfs.GMMFeaturesSelection()
model.learn_gmm(xt,yt)
idx, crit, [] = model.selection('forward',xt, yt,criterion='kappa', varNb=maxVar, nfold=5)

for i in range(maxVar):
    print "({0},{1})".format(wave[idx[i]],crit[i])

for i in range(maxVar):
    print "({0},{1})".format(i+1,crit[i])

# Save selected feature
D = sp.copy(model.mean[0,idx[:2]][:,sp.newaxis])

for i in xrange(1,9):
    D = sp.concatenate((D,model.mean[i,idx[:2]][:,sp.newaxis]),axis=1)

D = D.T
C = sp.arange(1,10)
D = sp.concatenate((D,C[:,sp.newaxis]),axis=1)
sp.savetxt("../FeatureExtraction/figures/fffsMean.csv",D,delimiter=',')
