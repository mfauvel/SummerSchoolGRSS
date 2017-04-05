import scipy as sp
import rasterTools as rt
import npfs as npfs
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Convenient fuctions
def compute_SVM(x,y,xt,yt,param_grid_svm):    
    y.shape=(y.size,)    
    cv = StratifiedKFold(n_splits=5,random_state=0).split(x,y)
    grid = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=cv,n_jobs=-1)
    grid.fit(x, y)
    clf = grid.best_estimator_
    clf.fit(x,y)
    yp = clf.predict(xt).reshape(yt.shape)
    return f1_score(yt,yp,average='weighted')

def compute_Linear_SVM(x,y,xt,yt,param_grid_svm):    
    y.shape=(y.size,)    
    cv = StratifiedKFold(n_splits=5,random_state=0).split(x,y)
    grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid_svm, cv=cv,n_jobs=-1)
    grid.fit(x, y)
    clf = grid.best_estimator_
    clf.fit(x,y)
    yp = clf.predict(xt).reshape(yt.shape)
    return f1_score(yt,yp,average='weighted')

def compute_RF(x,y,xt,yt,param_grid_rf):
    y.shape=(y.size,)    
    cv = StratifiedKFold(n_splits=5,random_state=0).split(x,y)
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_rf, cv=cv,n_jobs=-1)
    grid.fit(x, y)
    clf = grid.best_estimator_
    clf.fit(x,y)
    yp = clf.predict(xt).reshape(yt.shape)
    return f1_score(yt,yp,average='weighted')

def compute_FFFS(x,y,xt,yt,param_grid_fffs):
    maxVar = param_grid_fffs['maxvar']
    clf = npfs.GMMFeaturesSelection()
    clf.learn_gmm(x,y)
    idx, crit, [] = clf.selection('forward',x, y, criterion='F1Mean', varNb=maxVar, nfold=5)
    d_crit = sp.diff(crit)/crit[:-1]
    nv = sp.where(d_crit<param_grid_fffs['threshold'])[0][0]
    print("Number of variables {}".format(nv))
    yp = clf.predict_gmm(xt,featIdx=idx[:nv])[0]
    return f1_score(yt,yp,average='weighted')

def compute_KNN(x,y,xt,yt,param_grid_knn):
    y.shape=(y.size,)    
    cv = StratifiedKFold(n_splits=5,random_state=0).split(x,y)
    grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=param_grid_knn, cv=cv,n_jobs=-1)
    grid.fit(x, y)
    clf = grid.best_estimator_
    clf.fit(x,y)
    yp = clf.predict(xt).reshape(yt.shape)
    return f1_score(yt,yp,average='weighted')

if __name__ == '__main__':
    # Load data set
    X,y=rt.get_samples_from_roi('../Data/university.tif','../Data/university_gt.tif')
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1,random_state=0,stratify=y)

    # Parameters
    param_grid_svm = dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(0,3)) # SVM
    param_grid_linear_svm = dict(C=10.0**sp.arange(-2,3)) # LinearSVM
    param_grid_rf = dict(n_estimators=sp.arange(10,150,10)) # RF
    param_grid_fffs = dict(maxvar=20,threshold=0.001) # FFFS
    param_grid_knn = dict(n_neighbors = sp.arange(1,50,5))
    F1,CT=[],[]

    # Start the classification: SVM
    ts=time.time()
    F1.append(compute_SVM(X_train,y_train,X_test,y_test,param_grid_svm))
    CT.append(time.time()-ts)

    # Start the classification: RF
    ts=time.time()
    F1.append(compute_RF(X_train,y_train,X_test,y_test,param_grid_rf))
    CT.append(time.time()-ts)

    # Start the classification: LinearSVM
    ts=time.time()
    F1.append(compute_Linear_SVM(X_train,y_train,X_test,y_test,param_grid_linear_svm))
    CT.append(time.time()-ts)

    # Start the classification: FFFS
    ts=time.time()
    F1.append(compute_FFFS(X_train,y_train,X_test,y_test,param_grid_fffs))
    CT.append(time.time()-ts)

    # Start the classification: KNN
    ts=time.time()
    F1.append(compute_KNN(X_train,y_train,X_test,y_test,param_grid_knn))
    CT.append(time.time()-ts)
    
    # Print results
    print F1
    print CT
    for c in sp.unique(y_train):
        t = sp.where(y_train==c)[0]
        print("Number of training samples for class {0}:{1}".format(c,t.size))
    for c in sp.unique(y_train):
        t = sp.where(y_test==c)[0]
        print("Number of testing samples for class {0}:{1}".format(c,t.size))

    # Load data
    im,GeoT,Proj = rt.open_data('../Data/university.tif')
    [h,w,b]=im.shape
    im.shape=(h*w,b)
    im = sc.transform(im)
    
    # Perform the classification of the whole image
    y_train.shape=(y_train.size,)    
    cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
    grid = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=cv,n_jobs=-1)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    clf.fit(X_train,y_train)

    imp = clf.predict(im).reshape(h,w)
    
    # Save image
    rt.write_data('../Data/tm_university_svm.tif',imp,GeoT,Proj)
