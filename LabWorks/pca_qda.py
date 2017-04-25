# Labworks GRSS Summer School

# Initialization
import sys
sys.path.insert(0, '../Codes/')

# Do some usefull imports
import rasterTools as rt
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

# Parameters
NB = 50

# Load data
im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape
im.shape = (h*w,b)
ref =  rt.open_data('../Data/university_gt.tif')[0]
ref.shape = (h*w,)
idx = sp.where(ref>0)[0] # Get coordinate of the GT samples

# Apply PCA
pca = PCA()
imp = pca.fit_transform(im)
l = pca.explained_variance_
cl = (l.cumsum()/l.sum())[:NB]

# Split data
X_train, X_test, y_train, y_test = train_test_split(imp[idx,:], ref[idx], train_size=0.1,random_state=0,stratify=ref[idx])

# Classification
F1 = []
for i in xrange(1,NB+1):
    print("Number of PCs {}".format(i))
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train[:,:i],y_train)
    yp = clf.predict(X_test[:,:i]).reshape(y_test.shape)
    F1.append(f1_score(y_test,yp,average='weighted'))

# Plot accuracies function of the cummulative variance
plt.plot(cl[1:],F1[1:])
plt.axis([0.9, 1, 0.7, 1])
plt.show()
