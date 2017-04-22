from scipy import optimize
import scipy as sp
import rasterTools as rt
import pysptools.eea as eea

# Loss 
def loss(alpha,x,M):
    e = x-sp.dot(M,alpha)
    return (e**2).sum()

def jac(alpha,x,M):
    e = x-sp.dot(M,alpha)
    return -2*sp.dot(M.T,e)

cons = {'type':'eq','fun':lambda alpha: 1-alpha.sum(),'jac':lambda alpha: -alpha}
bnds = ((0, None), (0, None), (0, None,))

NE = 3
alpha0 = sp.ones((NE,))
alpha0 /= alpha0.sum()

# Load images
im,GeoT,Proj = rt.open_data('../Data/Moffett_full.tif')
[h,w,b]=im.shape
wave = sp.loadtxt('../Data/wave_moffett.csv',delimiter=',')

# Compute endmenbers
nfindr = eea.NFINDR()
M = nfindr.extract(im.astype(float), NE, normalize=False).T

abundances = sp.empty((h,w,NE))
for h_ in xrange(h):
    for w_ in xrange(w):
        x = im[h_,w_,:]
        # res = optimize.minimize(loss, alpha0, args=(x,M,), jac=jac,method='SLSQP', bounds=bnds,)
        # a = res['w']
        res = optimize.nnls(M,x)
        a = res[0]
        abundances[h_,w_,:]= (a/a.sum())

# Write the image
rt.write_data("../Data/Moffett_abundances.tif",abundances,GeoT,Proj)
