#ifndef C_UNIFORM_H
#define C_UNIFORM_H
#include <array>
#include <algorithm>

namespace demogreen {

    std::array<long int, 4> getNeighborhoodBounds(const unsigned int pixelIndex,
                                                  const unsigned int nbRows,
                                                  const unsigned int nbCols,
                                                  const unsigned int halfFilterSize);

    void applyUniform(float * inputColorImage,
                      float * outputFilteredImage,
                      unsigned int halfFilterSize,
                      unsigned int nbRows,
                      unsigned int nbCols,
                      unsigned int nbBands);

} // end of namespace demogreen

#endif