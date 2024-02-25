#ifndef IMAGE_H
#define IMAGE_H

#include <string>

/**
 * @brief Class representing an image.
 */
class Image {
private:
    int width;               /* Width of the image. */
    int height;              /* Height of the image. */
    int channels;            /* Number of color channels in the image. */
    unsigned char* imageData;/* Pointer to the image data. */
	std::string fileName;    /* Path to the image file. */

public:
    /**
     * @brief Default constructor for the Image class.
     */
    Image();

    /**
     * @brief Destructor for the Image class.
     */
    ~Image();

    /**
     * @brief Load an image from the specified file.
     * @param filename The path to the image file.
     * @return True if the image is loaded successfully, false otherwise.
     */
    bool loadImage(const char* filename);

    /**
     * @brief Save the image to the specified file.
     * @param fileName The path to the output image file.
     * @return True if the image is saved successfully, false otherwise.
     */
    bool saveImage(const char* fileName);

    /**
	* @brief Get the path to the image file.
	* @return Path to the image file.
    */
    std::string getFileName();
    
    /**
     * @brief Get the width of the image.
     * @return Width of the image.
     */
    int getWidth();

    /**
     * @brief Get the height of the image.
     * @return Height of the image.
     */
    int getHeight();

    /**
     * @brief Get the number of color channels in the image.
     * @return Number of color channels.
     */
    int getChannels();

    /**
     * @brief Get the pointer to the image data.
     * @return Pointer to the image data.
     */
    unsigned char* getImageData();

    /**
     * @brief Set the image data.
     * @param data Pointer to the new image data.
     */
    void setImageData(unsigned char* data);

    /**
     * @brief Get the total size of the image data.
     * @param includeCharSize If true, include the size of each char element in the calculation.
     * @return Total size of the image data.
     */
    size_t getTotalSize(bool includeCharSize = false);
};

#endif // !IMAGE_H