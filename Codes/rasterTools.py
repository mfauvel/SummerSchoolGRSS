# -*- coding: utf-8 -*-
import scipy as sp
from osgeo import gdal, ogr, osr

def open_data(filename):
    '''
    The function open and load the image given its name. 
    The type of the data is checked from the file and the scipy array is initialized accordingly.
    Input:
        filename: the name of the file
    Output:
        im: the data cube
        GeoTransform: the geotransform information 
        Projection: the projection information
    '''
    data = gdal.Open(filename,gdal.GA_ReadOnly)
    if data is None:
        print 'Impossible to open '+filename
        exit()
    nc = data.RasterXSize
    nl = data.RasterYSize
    d  = data.RasterCount
    
    # Get the type of the data
    gdal_dt = data.GetRasterBand(1).DataType
    if gdal_dt == gdal.GDT_Byte:
        dt = 'uint8'
    elif gdal_dt == gdal.GDT_Int16:
        dt = 'int16'
    elif gdal_dt == gdal.GDT_UInt16:
        dt = 'uint16'
    elif gdal_dt == gdal.GDT_Int32:
        dt = 'int32'
    elif gdal_dt == gdal.GDT_UInt32:
        dt = 'uint32'
    elif gdal_dt == gdal.GDT_Float32:
        dt = 'float32'
    elif gdal_dt == gdal.GDT_Float64:
        dt = 'float64'
    elif gdal_dt == gdal.GDT_CInt16 or gdal_dt == gdal.GDT_CInt32 or gdal_dt == gdal.GDT_CFloat32 or gdal_dt == gdal.GDT_CFloat64 :
        dt = 'complex64'
    else:
        print 'Data type unkown'
        exit()
    
    # Initialize the array
    if d ==1:
        im = sp.empty((nl,nc),dtype=dt)
        im =data.GetRasterBand(1).ReadAsArray()
    else:
        im = sp.empty((nl,nc,d),dtype=dt) 
        for i in range(d):
            im[:,:,i]=data.GetRasterBand(i+1).ReadAsArray()
    
    GeoTransform = data.GetGeoTransform()
    Projection = data.GetProjection()
    data = None
    return im,GeoTransform,Projection

def write_data(outname,im,GeoTransform,Projection):
    '''
    The function write the image on the  hard drive.
    Input: 
        outname: the name of the file to be written
        im: the image cube
        GeoTransform: the geotransform information 
        Projection: the projection information
    Output:
        Nothing --
    '''
    nl = im.shape[0]
    nc = im.shape[1]
    if im.ndim == 2:
        d=1
    else:
        d = im.shape[2]
    
    driver = gdal.GetDriverByName('GTiff')
    dt = im.dtype.name
    # Get the data type
    if dt == 'bool' or dt == 'uint8':
        gdal_dt=gdal.GDT_Byte
    elif dt == 'int8' or dt == 'int16':
        gdal_dt=gdal.GDT_Int16
    elif dt == 'uint16':
        gdal_dt=gdal.GDT_UInt16
    elif dt == 'int32':
        gdal_dt=gdal.GDT_Int32
    elif dt == 'uint32':
        gdal_dt=gdal.GDT_UInt32
    elif dt == 'int64' or dt == 'uint64' or dt == 'float16' or dt == 'float32':
        gdal_dt=gdal.GDT_Float32
    elif dt == 'float64':
        gdal_dt=gdal.GDT_Float64
    elif dt == 'complex64':
        gdal_dt=gdal.GDT_CFloat64
    else:
        print 'Data type non-suported'
        exit()
    
    dst_ds = driver.Create(outname,nc,nl, d, gdal_dt)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)
    
    if d==1:
        out = dst_ds.GetRasterBand(1)
        out.WriteArray(im)
        out.FlushCache()
    else:
        for i in range(d):
            out = dst_ds.GetRasterBand(i+1)
            out.WriteArray(im[:,:,i])
            out.FlushCache()
    dst_ds = None

def get_samples_from_roi(raster_name,roi_name):
    '''
    The function get the set of pixels given the thematic map. Both map should be of same size. Data is read per block.
    Input:
        raster_name: the name of the raster file, could be any file that GDAL can open
        roi_name: the name of the thematic image: each pixel whose values is greater than 0 is returned
    Output:
        X: the sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each 
            line of the matrix is a pixel.
        Y: the label of the pixel
    ''' 
    
    ## Open Raster
    raster = gdal.Open(raster_name,gdal.GA_ReadOnly)
    if raster is None:
        print 'Impossible to open '+raster_name
        exit()

    ## Open ROI
    roi = gdal.Open(roi_name,gdal.GA_ReadOnly)
    if roi is None:
        print 'Impossible to open '+roi_name
        exit()

    ## Some tests
    if (raster.RasterXSize != roi.RasterXSize) or (raster.RasterYSize != roi.RasterYSize):
        print 'Images should be of the same size'
        exit()

    ## Get block size
    band = raster.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    del band
    
    ## Get the number of variables and the size of the images
    d  = raster.RasterCount
    nc = raster.RasterXSize
    nl = raster.RasterYSize

    ## Read block data
    X,Y = [],[]
    for i in range(0,nl,y_block_size):
        if i + y_block_size < nl: # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in range(0,nc,x_block_size): # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            # Load the reference data
            ROI = roi.GetRasterBand(1).ReadAsArray(j, i, cols, lines)
            t = sp.nonzero(ROI)
            if t[0].size > 0:
                Y.extend(ROI[t].reshape((t[0].shape[0],1)).astype('uint8'))
                # Load the Variables
                Xtp = sp.empty((t[0].shape[0],d))
                for k in xrange(d):
                    band = raster.GetRasterBand(k+1).ReadAsArray(j, i, cols, lines)
                    Xtp[:,k] = band[t]
                try:
                    X.extend(Xtp)
                except MemoryError:
                    print 'Impossible to allocate memory: ROI too big'
                    exit()
    
    # Clean/Close variables
    del Xtp,band    
    roi = None # Close the roi file
    raster = None # Close the raster file

    return sp.asarray(X),sp.asarray(Y)

def get_samples_from_shp(raster_name, shape_name, Field="Class"):
    """
    The function get the set of pixels given a shapefile.
    Input:
    raster_name: the name of the raster file, could be any file that GDAL can open
    shape_name: the name of the shapefile
    Field: the name of the field that contains the labels
    Output:
        X: the sample matrix. A
            nXd matrix, where n is the number of referenced pixels and d is the number of variables. 
            Each line of the matrix is a pixel.
        Y: the label of the pixel
    """
    # Open Raster
    raster = gdal.Open(raster_name, gdal.GA_ReadOnly)
    if raster is None:
        print("Impossible to open {0}".format(raster_name))
        exit()

    # Open Shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_shp = driver.Open(shape_name, 0)
    if data_shp is None:
        print("Impossible to open {0}".format(shape_name))
        exit()
    layer = data_shp.GetLayer()

    # Some tests
    # Check if raster and shapefile overlap ?
    
    # Reproject vector geometry to same projection as raster
    layer_srs = layer.GetSpatialRef()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(layer_srs, raster_srs)

    # Get the number of variables and the size of the images, and geoinformation info
    d, nc, nl = raster.RasterCount, raster.RasterXSize, raster.RasterYSize
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]  # xmin
    yOrigin = transform[3]  # ymax
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    XMIN, XMAX, YMAX, YMIN = xOrigin, xOrigin+nc*pixelWidth, yOrigin, yOrigin+nl*pixelHeight
    
    # Iterate over features
    pi,Y=[],[]
    nf = layer.GetFeatureCount()

    # Get the coordinates of all pixels
    for f_ in xrange(nf):
        pi_, y_ = get_coords_from_polygon(layer.GetFeature(f_), coordTrans, XMIN, XMAX, YMIN, YMAX, pixelWidth, pixelHeight, Field, raster, layer, xOrigin, yOrigin, d, raster_srs)
        pi.extend(pi_)
        Y.extend(y_)

    # Read all spectral values
    ns = len(pi)
    X = sp.empty((ns, d))
    Y = sp.asarray(Y)

    for d_ in xrange(d):
        band = raster.GetRasterBand(d_+1).ReadAsArray()
        for p, pi_ in enumerate(pi):
            X[p, d_] = band[pi_[1], pi_[0]]

    # ### to be removed
    # temp = sp.zeros((nl,nc))
    # for pi_ in pi:
    #     temp[pi_[1],pi_[0]] = 1
    # write_data('test_feature.tif',temp,raster.GetGeoTransform(),raster.GetProjection())

    return X, Y

def get_coords_from_polygon(sample,coordTrans,XMIN,XMAX,YMIN,YMAX,pixelWidth,pixelHeight,Field,raster,layer,xOrigin,yOrigin,d,raster_srs):
    """
    """
    geom = sample.GetGeometryRef()
    geom.Transform(coordTrans)
    # TODO: Check if polygon or multipolygon
    ring = geom.GetGeometryRef(0)
    nbpoints = ring.GetPointCount()
    if (nbpoints > 0) & (sample.GetField(Field) > 0):  # Get all the vertices of the geometry
        pointsX, pointsY = [], []
        for p in range(nbpoints):
            lon, lat, z = ring.GetPoint(p)
            pointsX.append(lon)
            pointsY.append(lat)
        xmin,xmax,ymin,ymax=min(pointsX),max(pointsX),min(pointsY),max(pointsY)

        # Specify offset and rows and columns to read
        px,py = int((xmax - xmin)/pixelWidth)+1, int((ymin - ymax)/pixelHeight)+1

        # Check if the polygone is outside of the image
        xmin = (xmin if xmin >= XMIN else XMIN)
        xmax = (xmax if xmax <= XMAX else XMAX)
        ymin = (ymin if ymin >= YMIN else YMIN)
        ymax = (ymax if ymax <= YMAX else YMAX)

        # Create memory target raster
        mem_raster = gdal.GetDriverByName('MEM').Create('', px, py, 1, gdal.GDT_UInt16)
        mem_raster.SetGeoTransform((xmin, pixelWidth, 0,ymax, 0, pixelHeight,))

        # Create for target raster the same projection as for the value raster
        mem_raster.SetProjection(raster_srs.ExportToWkt())

        # Rasterize zone polygon to raster
        OPTIONS = 'ATTRIBUTE='+Field
        gdal.RasterizeLayer(mem_raster, [1],
                            layer, None,
                            options=[OPTIONS])

        # Get pixels for mem_raster
        t = sp.where(mem_raster.GetRasterBand(1).ReadAsArray()
                     == sample.GetField(Field))
        nt = t[0].size
        y = [sample.GetField(Field)]
        
        # Compute pixel coordinates
        m = [[xmin + pixelWidth*t[1][t_],
              ymax + pixelHeight*t[0][t_]] for t_ in xrange(nt)]
        pi = [[sp.floor((mx-xOrigin)/pixelWidth).astype(int),
               sp.floor((my-yOrigin)/pixelHeight).astype(int)]
              for mx, my in m]
        return pi, y*nt
    else:
        return [], []

