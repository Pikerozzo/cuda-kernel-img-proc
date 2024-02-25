#pragma once
#include <vector>
#include <string>
#include "FilterType.h"

/**
 * @brief Class representing a convolution kernel.
 */
class Kernel {
private:
    std::vector<float> kernelMatrix; /* Matrix representing the convolution kernel. */
    FilterType filterType;          /* Type of the convolution filter. */
    int kernelHeight;               /* Height of the convolution kernel matrix. */
    int kernelWidth;                /* Width of the convolution kernel matrix. */
    int filterNormalizationValue;   /* Normalization value for the filter. */

public:
    /**
     * @brief Default constructor for the Kernel class.
     * @param filterType Type of the convolution filter.
     */
    Kernel(FilterType filterType = FilterType::IDENTITY);

    /**
     * @brief Destructor for the Kernel class.
     */
    ~Kernel();

    /**
     * @brief Get the pointer to the convolution kernel matrix.
     * @return Pointer to the convolution kernel matrix.
     */
    const float* getKernelFilter();

    /**
     * @brief Get the height of the convolution kernel matrix.
     * @return Height of the convolution kernel matrix.
     */
    int getKernelHeight();

    /**
     * @brief Get the width of the convolution kernel matrix.
     * @return Width of the convolution kernel matrix.
     */
    int getKernelWidth();

    /**
     * @brief Get the normalization value for the filter.
     * @return Normalization value for the filter.
     */
    int getFilterNormalizationValue();

    /**
     * @brief Get the type of the convolution filter.
     * @return Type of the convolution filter.
     */
    FilterType getFilterType();

    /**
     * @brief Get the total size of the kernel data.
     * @param includeFloatSize If true, include the size of each float element in the calculation.
     * @return Total size of the kernel data.
     */
    size_t getTotalSize(bool includeFloatSize = false);

    /**
	 * @brief Set the filter convolution matrix.
	 * @param filterType Type of the convolution filter.
	 */
	void setFilter(FilterType filterType);    
};