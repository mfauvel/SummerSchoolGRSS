import scipy as sp
import rasterTools as rt
from sklearn.preprocessing import StandardScaler
import icm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Load data set
im,GeoT,Proj = rt.open_data('../Data/university.tif')
[h,w,b]=im.shape
im.shape=(h*w,b)

# Get the training set
X,y=rt.get_samples_from_roi('../Data/university.tif','../Data/university_gt.tif')

# Scale the data
sc = StandardScaler()
X = sc.fit_transform(X)
im = sc.transform(im)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05,random_state=0,stratify=y)

y_train.shape=(y_train.size,)    
cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
grid = GridSearchCV(SVC(), param_grid=dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(0,3)), cv=cv,n_jobs=-1)
grid.fit(X_train, y_train)
clf = grid.best_estimator_

clf.probability= True
clf.fit(X_train,y_train)

yp = clf.predict(X_test).reshape(y_test.shape)
print f1_score(y_test,yp,average='weighted')

del X_train, X_test, y_train, y_test

# Predict the whole image and the probability map
labels = clf.predict(im).reshape(h,w)
proba = -clf.predict_log_proba(im).reshape(h,w,y.max())

rt.write_data('../Data/proba_university_svm_proba.tif',proba,GeoT,Proj)
rt.write_data('../Data/proba_university_svm_labels.tif',labels,GeoT,Proj)

# Run ICM
diff = icm.fit(proba,labels,beta=1.25,th=0.01)
print diff
rt.write_data('../Data/tm_university_svm_mrf.tif',labels,GeoT,Proj)
