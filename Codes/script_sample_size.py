import scipy as sp
import rasterTools as rt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score

# Load data set
X,y=rt.get_samples_from_roi('../Data/university.tif','../Data/university_gt.tif')

# Split the data
X_, X_test, y_, y_test = train_test_split(X, y, train_size=0.50,random_state=0,stratify=y)

# Try differents size of the training set
SPLIT = sp.linspace(0.01,0.99,15)
F1,NS = sp.zeros_like(SPLIT),sp.zeros_like(SPLIT)
for i,split in enumerate(SPLIT):
    # Split the data
    X_train, _, y_train, _ = train_test_split(X_, y_, train_size=split,random_state=0,stratify=y_)
    # Learn the classifier
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    # Predict the classes
    yp = clf.predict(X_test)
    #Compute the F1
    F1[i],NS[i] = f1_score(y_test,yp,average='weighted'),y_train.size

D = sp.concatenate((NS[:,sp.newaxis],F1[:,sp.newaxis]),axis=1)
sp.savetxt("../Classification/figures/data_samples_size.csv",D,delimiter=',')
