import scipy as sp
import rasterTools as rt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score

# Load data set
X,y=rt.get_samples_from_roi('../Data/university.tif','../Data/university_gt.tif')

# Split the data
X_, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25,random_state=0,stratify=y)

# Try differents size of the training set
SKIP = sorted(range(1,11),reverse=True)
FS,NF = sp.zeros_like(SKIP,dtype='float'),sp.zeros_like(SKIP)
for i,skip in enumerate(SKIP):
    # Skip some variables
    X_train =X_[:,::skip]
    # Learn the classifier
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    # Predict the classes
    yp = clf.predict(X_test[:,::skip])
    #Compute the FS
    FS[i], NF[i] = f1_score(y_test,yp,average='weighted'), X_train.shape[1]

D = sp.concatenate((NF[:,sp.newaxis],FS[:,sp.newaxis]),axis=1)
sp.savetxt("../Classification/figures/data_features_size.csv",D,delimiter=',')
