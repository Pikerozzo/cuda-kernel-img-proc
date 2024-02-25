#include <stdio.h>
#include <chrono>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>

#include "Kernel.h"
#include "Image.h"
#include "FilterType.h"
#include "Convolution.cuh"

using namespace std;

/**
* @brief Format a duration in microseconds to a string
* @return Formatted microseconds in string
*/
string formatMicroseconds(chrono::microseconds duration)
{
	auto seconds = chrono::duration_cast<chrono::seconds>(duration);
	auto millis = chrono::duration_cast<chrono::milliseconds>(duration % chrono::seconds(1));
	auto micros = duration % chrono::milliseconds(1);

	stringstream ss;
	ss << seconds.count();
	string secondsStr = ss.str();
	ss.str("");
	ss << setw(3) << setfill('0') << millis.count();
	string millisStr = ss.str();
	ss.str("");
	ss << setw(3) << micros.count();
	string microsStr = ss.str();

	return secondsStr + "'" + millisStr + "'" + microsStr;
}


/**
*
* ##	Kernel Image Processing with CUDA	##
*
* This project focused on the implementation of a convolution operation using CUDA.
* The project was developed as part of the Parallel Programming for Machine Learning course at the University of Florence.
*
* Gregorio Piqué
*/
int main(int argc, char* argv[])
{

	Image image{};
	string filterTypeParam = "box_blur";
	Kernel kernelFilter{ FilterType::BOX_BLUR };

	bool isParallel = true;
	string execModeParam = "constant";
	ExecutionMode execMode = ExecutionMode::CONSTANT;

	const char* inputFileName = "..\\imgs\\04_bryce.jpg";
	const char* outputFileName = "..\\output\\result.png";

	if (argc > 1) {
		inputFileName = argv[1];

		if (argc > 2)
		{
			filterTypeParam = argv[2];
			if (filterTypeParam == "box_blur")
			{
				kernelFilter.setFilter(FilterType::BOX_BLUR);
			}
			else if (filterTypeParam == "edge")
			{
				kernelFilter.setFilter(FilterType::EDGE_DETECTION);
			}
			else if (filterTypeParam == "gaussian")
			{
				kernelFilter.setFilter(FilterType::GAUSSIAN_BLUR);
			}
			else if (filterTypeParam == "h_emboss")
			{
				kernelFilter.setFilter(FilterType::H_EMBOSS);
			}
			else if (filterTypeParam == "identity")
			{
				kernelFilter.setFilter(FilterType::IDENTITY);
			}
			else if (filterTypeParam == "v_emboss")
			{
				kernelFilter.setFilter(FilterType::V_EMBOSS);
			}
			else if (filterTypeParam == "sharpen")
			{
				kernelFilter.setFilter(FilterType::SHARPEN);
			}
			else
			{
				cerr << "Error! \"" << filterTypeParam << "\" is not a valid filter type. Allowed values are: < box_blur | edge | gaussian | h_emboss | identity | v_emboss | sharpen >. Using default filter settings." << endl;
				filterTypeParam = "box_blur";
			}

			if (argc > 3)
			{
				execModeParam = argv[3];
				if (execModeParam == "constant")
				{
					execMode = ExecutionMode::CONSTANT;
				}
				else if (execModeParam == "global")
				{
					execMode = ExecutionMode::GLOBAL;
				}
				else if (execModeParam == "shared")
				{
					execMode = ExecutionMode::SHARED;
				}
				else if (execModeParam == "sequential")
				{
					execMode = ExecutionMode::SEQUENTIAL;
					isParallel = false;
				}
				else
				{
					cerr << "Error! \"" << execModeParam << "\" is not a valid execution mode. Allowed values are: < constant | global | shared | sequential >. Using default execution mode settings." << endl;
					execModeParam = "constant";
				}
			}
		}
	}
	else {
		cout << "No additional parameters specified: using default configuration." << endl;
	}
	cout << endl;

	if (!image.loadImage(inputFileName)) {
		cerr << "Error: Could not load image. Terminating." << endl;
		return 1;
	}

	cout << "Config settings" << endl;
	cout << "\tImage name:     \t" << image.getFileName() << endl;
	cout << "\tImage size:     \t" << image.getWidth() << " x " << image.getHeight() << endl;
	cout << "\tFilter type:	   \t" << filterTypeParam << endl;
	cout << "\tExecution mode: \t" << (isParallel ? "parallel - " : "") << execModeParam << (isParallel ? " memory" : "") << endl << endl;


	Convolution convolution{};

	cout << "Image processing starting... ";
	auto start = chrono::high_resolution_clock::now();

	convolution.apply(image, kernelFilter, execMode);

	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "Done." << endl;
	cout << "Execution time : " << formatMicroseconds(duration) << " microseconds" << endl;


	ofstream resultsFile;
	resultsFile.open("..\\results\\exec.csv");
	if (!resultsFile.is_open()) {
		cerr << "Error opening file: .\\results\\exec.csv" << endl;
	}
	else {		
		resultsFile << "img_name,img_id,img_w,img_h,filter,mode,exec_time\n";
		string c = "\\";
		auto startPos = image.getFileName().find_last_of(c);
		auto file_name = image.getFileName().substr(startPos + c.length(), image.getFileName().length() - startPos);
		resultsFile << file_name << "," << image.getWidth() << "," << image.getHeight() << "," << filterTypeParam << "," << execModeParam << "," << duration.count() << "\n";
	}

	resultsFile.close();



	cout << "Saving image... ";
	string outputFileNameStr = string(outputFileName);
	auto ext = outputFileNameStr.find_last_of(".");
	if (ext != string::npos)
	{
		outputFileNameStr = outputFileNameStr.insert(ext, "_" + filterTypeParam);
	}


	if (image.saveImage(outputFileNameStr.c_str()))
	{
		cout << "Done." << endl;
	}
	else
	{
		cout << "Error: Image was not saved successfully." << endl;
		return 1;
	}

	convolution.resetCuda(execMode);

	return 0;
}