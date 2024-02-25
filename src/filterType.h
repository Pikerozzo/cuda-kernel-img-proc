#ifndef FILTERTYPE_H
#define FILTERTYPE_H
#include <string>

/**
 * @brief Different types of image filters.
 */
enum class FilterType {
    BOX_BLUR,           /* Box blur filter. */
    EDGE_DETECTION,     /* Edge detection filter. */
    GAUSSIAN_BLUR,      /* Gaussian blur filter. */
    H_EMBOSS,           /* Horizontal emboss filter. */
    IDENTITY,           /* Identity filter. */
    V_EMBOSS,           /* Vertical emboss filter. */
    SHARPEN             /* Sharpen filter. */
};
#endif // !FILTERTYPE_H