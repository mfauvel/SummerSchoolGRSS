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
from sklearn.model_selection import cross_val_score

# Number of Principal components to test
NB = 40

# Load data and the ground truth 
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

# Classification and estimation of the accuracies
F1t,F1v,F1e = [],[],[]
for i in xrange(1,NB+1):
    print("Number of PCs {}".format(i))
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train[:,:i],y_train)
    # Classify the training samples
    yp = clf.predict(X_train[:,:i]).reshape(y_train.shape)
    F1t.append(f1_score(y_train,yp,average='weighted'))
    # Classify the testing samples
    yp = clf.predict(X_test[:,:i]).reshape(y_test.shape)
    F1v.append(f1_score(y_test,yp,average='weighted'))
    # Estimation of the classification with CV
    scores = cross_val_score(clf, X_train[:,:i], y_train, cv=5, scoring='f1_weighted')
    F1e.append(scores.mean())
    
# Plot accuracies function of the cummulative variance
nf = sp.argmax(F1e)
plt.plot(cl,F1t,label='Training set',linewidth=3)
plt.plot(cl,F1v,label='Validation set',linewidth=3)
plt.plot(cl,F1e,label='Cross validation',linewidth=3)
plt.axis([0.98, 1, 0.7, 1])
plt.xlabel("Percentage of cummulative variance")
plt.ylabel("F1 score")
plt.legend(loc='upper center', shadow=True)
plt.title("Number of principal features {} corresponding to {} % of the cummulative variance".format(nf+1,cl[nf]))
plt.grid(True)
plt.show()
