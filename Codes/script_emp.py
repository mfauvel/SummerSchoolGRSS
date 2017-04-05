import scipy as sp
from skimage.morphology import disk, erosion, dilation, reconstruction
import rasterTools as rt

def morphological_profile(im,radius=1,step=2,no=4):
    """ Compute the morphological profile of a given flat image with a disk as structuring element
    INPUT:
    im: input image, must be flat
    radius: initial size of SE
    step: step size for the SE
    no: number of opening/closing
    OUTPUT:
    MP: morphological profile, image of size h*w*(2*no+1)
    """
    if im.ndim != 2:
        print("Image should be flat")
        exit()

    # Initialization of the output
    [h,w] = im.shape
    out = sp.empty((h,w,2*no+1))
    out[:,:,no]=im.copy()

    # Start the computation
    for i in xrange(no):
        # Structuring elements
        se = disk(radius+i*2)

        # Compute opening per reconstruction
        temp = erosion(im,se)
        out[:,:,no+1+i] = reconstruction(temp,im,method='dilation')

        # Compute closing per reconstruction
        temp = dilation(im,se)
        out[:,:,no-1-i] = reconstruction(temp,im,method='erosion')

    return out

if __name__ == '__main__':
    # Load image
    im,GeoT,Proj = rt.open_data('../Data/pca_university.tif')

    # Apply the Morphological profile on each PC
    EMP = []
    for i in xrange(3):
        EMP.append(morphological_profile(im[:,:,i]))
    EMP = sp.concatenate(EMP,axis=2)
    rt.write_data("../Data/emp_pca_university.tif",EMP,GeoT,Proj)
