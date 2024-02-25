#include "Kernel.h"


Kernel::Kernel(FilterType filterType)
{
	setFilter(filterType);
}

Kernel::~Kernel() = default;

const float* Kernel::getKernelFilter()
{
	return const_cast<float*>(kernelMatrix.data());
}

int Kernel::getKernelHeight()
{
	return kernelHeight;
}

int Kernel::getKernelWidth()
{
	return kernelWidth;
}

int Kernel::getFilterNormalizationValue()
{
	return filterNormalizationValue;
}

FilterType Kernel::getFilterType()
{
	return filterType;
}

size_t Kernel::getTotalSize(bool includeFloatSize) {
	if (includeFloatSize)
	{
		return kernelHeight * kernelWidth * sizeof(float);
	}
	return kernelHeight * kernelWidth;
}

void Kernel::setFilter(FilterType filterType)
{
	kernelHeight = 3;
	kernelWidth = 3;
	filterNormalizationValue = 0;

	switch (filterType)
	{
	case FilterType::EDGE_DETECTION:
		kernelMatrix = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
		filterType = FilterType::EDGE_DETECTION;
		break;
	case FilterType::BOX_BLUR:
		kernelMatrix = { 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0 };
		filterType = FilterType::BOX_BLUR;
		break;
	case FilterType::GAUSSIAN_BLUR:
		kernelMatrix = { 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0 };
		filterType = FilterType::GAUSSIAN_BLUR;
		break;
	case FilterType::H_EMBOSS:
		kernelMatrix = { 0, 0, 0, 1, 0, -1, 0, 0, 0 };
		filterNormalizationValue = 128;
		filterType = FilterType::H_EMBOSS;
		break;
	case FilterType::V_EMBOSS:
		kernelMatrix = { 0, 1, 0, 0, 0, 0, 0, -1, 0 };
		filterNormalizationValue = 128;
		filterType = FilterType::V_EMBOSS;
		break;
	case FilterType::SHARPEN:
		kernelMatrix = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
		filterType = FilterType::SHARPEN;
		break;
	default:
	case FilterType::IDENTITY:
		kernelMatrix = { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
		filterType = FilterType::IDENTITY;
		break;
	}
}