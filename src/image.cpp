#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

Image::Image()
{
	width = 0;
	height = 0;
	channels = 0;
	imageData = nullptr;
	fileName = "";
}

Image::~Image()
{
	if (imageData != nullptr)
	{
		stbi_image_free(imageData);
	}
}

bool Image::loadImage(const char* filename)
{
	imageData = stbi_load(filename, &width, &height, &channels, 0);
	if (imageData == nullptr)
	{
		return false;
	}
	this->fileName = filename;
	
	return true;
}

bool Image::saveImage(const char* fileName)
{
	return stbi_write_png(fileName, width, height, channels, imageData, width * channels) == 1;
}

std::string Image::getFileName() {
	return fileName;
}

int Image::getWidth()
{
	return width;
}

int Image::getHeight()
{
	return height;
}

int Image::getChannels()
{
	return channels;
}

unsigned char* Image::getImageData()
{
	return imageData;
}

void Image::setImageData(unsigned char* data)
{
	stbi_image_free(imageData);
	imageData = data;
}

size_t Image::getTotalSize(bool includeCharSize)
{
	if (includeCharSize)
	{
		return width * height * channels * sizeof(unsigned char);
	}
	
	return width * height * channels;
}