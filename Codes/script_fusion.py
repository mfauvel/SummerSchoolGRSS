import rasterTools as rt
import scipy as sp
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from script_emp import morphological_profile
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Load data set
im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape
im.shape=(h*w,b)

# Compute the morphological profile
pca = PCA(n_components=3)
pcs = pca.fit_transform(im)
EMP = []
for i in xrange(3):
    EMP.append(morphological_profile(pcs[:,i].reshape(h,w),step=1,no=10))
EMP = sp.concatenate(EMP,axis=2)
EMP.shape=(h*w,EMP.shape[2])
del pcs

# Concatenate the spectral and spatial features and do scaling
IM_EMP = sp.concatenate((im,EMP.astype(im.dtype)),axis=1)

del im,EMP

# Save the results
rt.write_data("../Data/fusion_inputs_university.tif",IM_EMP.reshape(h,w,IM_EMP.shape[1]),GeoT,Proj)

# Get the training set
X,y=rt.get_samples_from_roi('../Data/fusion_inputs_university.tif','../Data/university_gt.tif')

# Scale the data
sc = StandardScaler()
X = sc.fit_transform(X)
IM_EMP = sc.transform(IM_EMP)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1,random_state=0,stratify=y)

y_train.shape=(y_train.size,)    
cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
grid = GridSearchCV(SVC(), param_grid=dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(0,3)), cv=cv,n_jobs=-1)
grid.fit(X_train, y_train)
clf = grid.best_estimator_
clf.fit(X_train,y_train)
yp = clf.predict(X_test).reshape(y_test.shape)
print f1_score(y_test,yp,average='weighted')

del X_train, X_test, y_train, y_test
# Predict the whole image
imp = clf.predict(IM_EMP)
rt.write_data('../Data/tm_university_fusion.tif',imp.reshape(h,w),GeoT,Proj)
