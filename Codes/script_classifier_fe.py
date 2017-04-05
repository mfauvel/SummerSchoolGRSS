import scipy as sp
import rasterTools as rt
from sklearn import neighbors
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score

DATA = ['../Data/university.tif','../Data/pca_university.tif','../Data/lda_university.tif',
        '../Data/kpca_university.tif']
GT = '../Data/university_gt.tif'

F1_knn,F1_gmm = [],[]
for data in DATA:
    print data
    # Load data set
    X,y=rt.get_samples_from_roi(data,GT)
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1,random_state=0,stratify=y)

    # Compute Cross validation for knn
    y_train.shape=(y_train.size,)
    cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,y_train)
    grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=dict(n_neighbors = sp.arange(1,50,5)), cv=cv,n_jobs=-1)
    grid.fit(X_train, y_train)

    # Compute classification for knn
    clf = grid.best_estimator_
    clf.fit(X_train,y_train)
    yp = clf.predict(X_test).reshape(y_test.shape)
    F1_knn.append(f1_score(y_test,yp,average='weighted'))

    # Compute classification for GMM
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    yp = clf.predict(X_test)
    F1_gmm.append(f1_score(y_test,yp,average='weighted'))
    
    # Clean data
    X,X_train,X_test,y,y_train,y_test=[],[],[],[],[],[]

print F1_knn
print F1_gmm
