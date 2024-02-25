#include "Kernel.h"
#include "Image.h"
#include "ExecutionMode.h"

/**
 * @class Convolution
 *
 * Class that represents and implements the convolution operation
 */
class Convolution {
public:
	/**
	 * @brief Constructor and destructor
	 */
	Convolution();
	~Convolution();

	/**
	 * @brief Apply the convolution operation to the image using the specified kernel and execution mode
	 *
	 * @param image Image to apply the convolution operation to
	 * @param kernel Kernel filter to use for the convolution operation
	 * @param execMode Execution mode to use for the convolution operation
	 */
	void apply(Image& image, Kernel& kernel, ExecutionMode execMode = ExecutionMode::GLOBAL);
	
	/**
	 * @brief Reset the CUDA device
	 */
	void resetCuda(ExecutionMode);

private:
	/**
	 * @brief Apply the convolution to the image using the specified kernel, while using constant memory for the kernel filter and global memory for the image
	 * 
	 * @param image Image to apply the convolution operation to
	 * @param kernel Kernel filter to use for the convolution operation
	 */
	void applyConstant(Image& image, Kernel& kernel);

	/**
	 * @brief Apply the convolution to the image using the specified kernel, while using global memory for both the kernel filter and the image
	 *
	 * @param image Image to apply the convolution operation to
	 * @param kernel Kernel filter to use for the convolution operation
	 */
	void applyGlobal(Image& image, Kernel& kernel);

	/**
	 * @brief Apply the convolution to the image using the specified kernel, while using constant memory for the kernel filter and shared memory for loading tiles of the image
	 *
	 * @param image Image to apply the convolution operation to
	 * @param kernel Kernel filter to use for the convolution operation
	 */
	void applyShared(Image& image, Kernel& kernel);

	/**
	 * @brief Apply the convolution to the image using the specified kernel with a sequential approach
	 *
	 * @param image Image to apply the convolution operation to
	 * @param kernel Kernel filter to use for the convolution operation
	 */
	void applySequential(Image& image, Kernel& kernel);
};