import scipy as sp
import rasterTools as rt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Convenient Class for summation kernel
class CompositeKernel(BaseEstimator,TransformerMixin):
    def __init__(self,mu=0.5,gamma=1.0):
        self.gamma = gamma
        self.mu = mu
        
    def transform(self,X):
        K = self.mu*rbf_kernel(X[:,:-3],self.Xs_,gamma=self.gamma)
        K += (1-self.mu)*rbf_kernel(X[:,-3:],self.Xw_,gamma=self.gamma)
        return K

    def fit(self,X,y=None, **fit_params):
        self.Xs_ = X[:,:-3]
        self.Xw_ = X[:,-3:]
        return self
    
# Load data
Xs,y = rt.get_samples_from_roi('../Data/university.tif','../Data/university_gt.tif')
Xw,y = rt.get_samples_from_roi('../Data/pca_median_11_11_university.tif','../Data/university_gt.tif')
scs = StandardScaler()
Xs = scs.fit_transform(Xs)
scw = StandardScaler()
Xw = scw.fit_transform(Xw)

# Split data
Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, train_size=0.05,random_state=0,stratify=y)
Xw_train, Xw_test, y_train, y_test = train_test_split(Xw, y, train_size=0.05,random_state=0,stratify=y)
y_train.shape=(y_train.size,) 
X_train = sp.concatenate((Xs_train,Xw_train),axis=1)
X_test = sp.concatenate((Xs_test,Xw_test),axis=1)
print X_train.shape

# Create a pipeline
pipe = Pipeline([
    ('CK',CompositeKernel()),
    ('SVM',SVC())
])

# Optimize parameters
cv_params = dict([
    ('CK__gamma', 2.0**sp.arange(-3,3)),
    ('CK__mu', sp.linspace(0,1,num=11)),
    ('SVM__kernel', ['precomputed']),
    ])

cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
grid = GridSearchCV(pipe, cv_params, cv=cv, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
print grid.best_params_
clf = grid.best_estimator_
clf.fit(X_train, y_train)
yp = clf.predict(X_test)
print f1_score(y_test,yp,average='weighted')

# Load image
ims,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=ims.shape
ims.shape=(h*w,b)
imw,GeoT,Proj = rt.open_data('../Data/pca_median_11_11_university.tif')
[h,w,b]=imw.shape
imw.shape=(h*w,b)
ims = scs.transform(ims)
imw = scw.transform(imw)
im = sp.concatenate((ims,imw),axis=1)
del imw, ims, X_train, X_test, Xs_train, Xs_test, Xw_train, Xw_test,
imp = clf.predict(im)
rt.write_data('../Data/tm_university_ck_mw.tif',imp.reshape(h,w),GeoT,Proj)
