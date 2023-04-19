# distutils: language = c++
import numpy as np

# Begin PXD section 
# (similar to C header files to expose C++ methods that will be used by our cython module)

# Necessary to include the C++ code from PLKMeans
cdef extern from "cppsrc/c_uniform.cpp":
    pass

cdef extern from "cppsrc/c_uniform.h" namespace "demogreen":
    
    void applyUniform(float * inputColorImage,
                      float * outputFilteredImage,
                      unsigned int halfFilterSize,
                      unsigned int nbRows,
                      unsigned int nbCols,
                      unsigned int nbBands)

# End PXD section

# Begin PYX section


def npAsContiguousArray(arr):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr) # Makes a contiguous copy of the numpy array.
    return arr


def cyUniformFilter(colorImgBuffer: np.ndarray,
                    halfFilterSize: int) -> np.ndarray:
    """ Cython version of the uniform filter """
    
    # Retrieve dimension of the input image
    nbBands: int = colorImgBuffer.shape[0]
    nbRows: int = colorImgBuffer.shape[1]
    nbCols: int = colorImgBuffer.shape[2]

    # Turns the color input image buffer to a memory view python object that provides direct access
    # to the contiguous buffer memory
    cdef float [::1] input_img_memview = npAsContiguousArray(colorImgBuffer.flatten().astype(np.float32))
    
    # Allocate the output
    cdef float [::1] output_img_memview = np.zeros( (nbBands * nbRows * nbCols), dtype=np.float32, order='C' )

    applyUniform(&input_img_memview[0],
                 &output_img_memview[0],
                 halfFilterSize,
                 nbRows,
                 nbCols,
                 nbBands)
    
    return np.resize(np.asarray(output_img_memview), (nbBands, nbRows, nbCols))


# End PYX section


