# Labworks GRSS Summer School

# Initialization
import sys
sys.path.insert(0, '../Codes/')

# Do some usefull import
import scipy as sp
import rasterTools as rt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#########################################################################################

#################################
## Spectral Feature Extraction ##
#################################

# Do you need/want to do spectral feature extraction ?
# See one of the following files and copy/paste what you need:
# ../Codes/fffsPavia.py
# ../Codes/pcaPavia.py
# ../Codes/kpcaPavia.py
# ../Codes/ldaPavia.py

################################
## Spatial Feature Extraction ##
################################

# Do you want to do spatial feature extraction ?
# See one the following files and copy/past what you need
# ../Codes/script_emp.py
# 
# You can also see these links for template filtering
# http://scikit-image.org/docs/stable/api/skimage.filters.html#median
# http://scikit-image.org/docs/stable/api/skimage.filters.rank.html#entropy
# http://scikit-image.org/docs/stable/api/skimage.filters.rank.html#mean

#####################################
## Data Fusion or Composite kernel ##
#####################################

# You can perform feature fusion, classifier fusion and combine feature with composite kernel
# See one of the following files and copy/past what you need:
# ../Codes/script_CK_mw.py
# ../Codes/script_classifier.py
# ../Codes/script_fusion.py

#################################
## Spatial Post Regularization ## 
#################################

# You can do a post-classification spatial regularization using voting scheme or MRF
# script_fusion_mv.py
# script_mrf.py

#########################################################################################

# Baseline

# Load data set
X,y=rt.get_samples_from_roi('92AV3C.bil','gt.tif')
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1,random_state=0,stratify=y)
print X_train.shape

# Learn SVM
param_grid_svm = dict(gamma=2.0**sp.arange(-8,2), C=10.0**sp.arange(0,3)) # SVM
y_train.shape=(y_train.size,)    
cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
grid = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=cv,n_jobs=-1)
grid.fit(X_train, y_train)
print grid.best_params_
clf = grid.best_estimator_
clf.fit(X_train,y_train)
yp = clf.predict(X_test).reshape(y_test.shape)


# Classify the whole image
im,GeoT,Proj = rt.open_data('92AV3C.bil')
[h,w,b]=im.shape
im.shape=(h*w,b)
im = sc.transform(im)
imp = clf.predict(im).reshape(h,w)
rt.write_data('thematic_map.tif',imp,GeoT,Proj)

# Check the accuracy
yp,y=rt.get_samples_from_roi('thematic_map.tif','gt.tif')
print f1_score(yp,y,average='weighted')
