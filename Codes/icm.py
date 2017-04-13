import scipy as sp
import rasterTools as rt

# Convenient functions
def compute_energy(proba,classes,beta):
    """
    The function compute the spatial energy terms of the Potts model
    classes: a 3x3 array containing the labels, the considered pixels is in the "middle" classes[1,1]
    proba:  the conditional probabilities of the considered pixels
    beta: the weight parameter
    """

    # Potts model
    count = (classes!=classes[1,1]).sum()

    # Add spectral and spatial energy
    energy = proba + beta*count

    return energy

# Main function
def fit(proba,labels,beta=4,th=0.000001):
    """
    """
    # Get some parameters and do initialization
    diff = [1]
    niter = 0
    [nl,nc,C]=proba.shape

    # Iterate until convergence
    while (diff[-1] > th) and (niter < 100):
        old_labels= labels.copy() # Make a copy of the old labels
        for i in xrange(1,nl-1): # Scan each line
            for j in xrange(1,nc-1): # Scan each column
                energy = []
                labels_ = old_labels[i-1:i+2,j-1:j+2].copy()
                for c in xrange(C): # Compute the energy for the different classes
                    labels_[1,1] = c+1
                    energy.append(compute_energy(proba[i,j,c],labels_,beta))
                arg = sp.argmin(energy) # Get the maximum energy term for the local configuration
                labels[i,j] = arg + 1
        diff.append(1 - sp.sum(old_labels == labels ).astype(float)/nc/nl) # Compute the changes
        niter += 1
    # Clean data
    del old_labels
    return diff
