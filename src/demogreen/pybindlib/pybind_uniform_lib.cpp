#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <array>
#include <algorithm>

/*
    Pure C++ code
*/
std::array<long int, 4> getNeighborhoodBounds(const unsigned int pixelIndex,
                                                const unsigned int nbRows,
                                                const unsigned int nbCols,
                                                const unsigned int halfFilterSize) {

    // array of 4 bounds: [ startX, startY, endX, endY ]
    std::array<long int, 4> bounds;

    const long int pixelCol = pixelIndex % nbCols;
    const long int pixelRow = pixelIndex / nbCols;

    // StartX
    bounds[0] = std::max((long int)0, pixelCol - halfFilterSize);
    // StartY
    bounds[1] = std::max((long int)0, pixelRow - halfFilterSize);
    // EndX
    bounds[2] = std::min((long int)nbCols - 1, pixelCol + halfFilterSize);
    // EndY
    bounds[3] = std::min((long int)nbRows - 1, pixelRow + halfFilterSize);

    return bounds;
}



/*
    Python interface
*/

namespace py = pybind11;


void pbApplyUniform(py::array_t<float>& pyInputColorImage,
                    py::array_t<float>& pyOutputFilteredImage,
                    unsigned int halfFilterSize,
                    unsigned int nbRows,
                    unsigned int nbCols,
                    unsigned int nbBands) {
        
    // input color image are flattened and values are organised in a contiguous way 1-D array in memory
    // [Red values, Green Values, Blue Values]

    // Get info from input color image and
    // and output filtered image
    py::buffer_info buf1 = pyInputColorImage.request();
    py::buffer_info buf2 = pyOutputFilteredImage.request();

    if (buf1.ndim !=1 || buf2.ndim !=1)
    {
        throw std::runtime_error("Number of dimensions must be one");
    }

    if (buf1.size !=buf2.size)
    {
        throw std::runtime_error("Input shape must match");
    }

    // Like this we avoid memory duplication
    float * inputColorImage = (float*) buf1.ptr;
    float * outputFilteredImage = (float*) buf2.ptr;

    const unsigned int nbPixelsPerChannel = nbRows * nbCols;
    float meanValue = 0.f;

    for(unsigned int p = 0; p < nbPixelsPerChannel; p++){

        auto bounds = getNeighborhoodBounds(p,
                                            nbRows,
                                            nbCols,
                                            halfFilterSize);
        
        // How many pixels contribute to the mean value
        const float den = (bounds[3] - bounds[1] + 1) * (bounds[2] - bounds[0] + 1);

        // Compute mean for each spectral band
        for(unsigned int b = 0; b < nbBands; b++){

            // Reset the mean value
            meanValue = 0.f;
            
            for(unsigned int rr =  bounds[1]; rr <= bounds[3]; rr++){
                for(unsigned int cc = bounds[0]; cc <= bounds[2]; cc++){
                    meanValue += inputColorImage[ b * nbPixelsPerChannel + rr * nbCols + cc ];
                }
            }

            outputFilteredImage[ b * nbPixelsPerChannel + p ] = meanValue / den;
        }
    }
}

// wrap as Python module
PYBIND11_MODULE(pbuniform, m)
{
    m.doc() = "pbuniform";

    m.def("pbApplyUniform", &pbApplyUniform, "Apply uniform filter on a satellite image",
	py::arg("pyInputColorImage"),
	py::arg("pyOutputFilteredImage"),
	py::arg("halfFilterSize"),
	py::arg("nbRows"),
	py::arg("nbCols"),
	py::arg("nbBands")
	);
}
