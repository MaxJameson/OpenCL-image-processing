#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;


void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {

	// Get starting timepoint
	auto Mainstart = std::chrono::high_resolution_clock::now();
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {

		bool depthSelect = false;
		int depth;

		// loops until a valid bin is input
		while(!depthSelect) {

			// takes input for bin
			std::cout << "Please enter the bit depth of the image: 8 or 16: "; // Type a number and press enter
			std::cin >> depth; // Get user input from the keyboard

			// checsk if input is valid
			if (depth == 8 || depth == 16) {
				depthSelect = true;
				std::cout << depth << " bit selected" << endl;
			}
			else {
				std::cout << "Please select a valid bit depth" << endl;
				std::cin.clear();
				std::cin.ignore(1, '\n');
			}

		}

		CImg<unsigned char> image_input(image_filename.c_str());

		//CImg<unsigned short> input(image_filename.c_str());

		//CImgDisplay tester(input, "output");

		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		std::cout << "" << endl;

		////////////////////////////////////////////////////////
		/////////////// Image and bin formatting
		////////////////////////////////////////////////////////

		// create a vector to store pixels
		std::vector<unsigned int> pixels;
		// stores end of intensity values for colour images
		std::vector<unsigned int> intenEnd;

		// checks if the image is colour of greysacel
		if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {

			// converts colour space
			image_input = image_input.RGBtoYCbCr();


			// creates vector of intensity values and vector of chroma red + blue
			pixels.assign(image_input.begin(), image_input.begin() + (image_input.size() / 3));
			intenEnd.assign(image_input.begin() + (image_input.size() / 3) + 1, image_input.end());

		}
		else {

			// creates vector of greyscale intensities
			pixels.assign(image_input.begin() + 0, image_input.begin() + image_input.size());
		}
			 
		// stores bool to check if the value of bins is valid
		bool binCheck = false;
		// stores number of bins
		unsigned int bins;
		// stores number of bits per pixel
		unsigned int bits = 256;
		// stores valid to calculate which bin a pixel belongs too
		unsigned int binsDivider;

		// loops until a valid bin is input
		while (!binCheck) {

			// takes input for bin
			std::cout << "Please enter the number of bins you would like between the range of 32 - 256: "; // Type a number and press enter
			std::cin >> bins; // Get user input from the keyboard

			// checsk if input is valid
			if (bins < 32 || bins > 256) {
				std::cout << "Invalid input " << endl;
				std::cin.clear();
				std::cin.ignore(1, '\n');
			}
			else {
				binCheck = true;

				// calculates the number used to define which bin and intensity belongs too
				if (bins = bits) {
					binsDivider = 1;
				}
				else {
					binsDivider = bits / bins;
				}
			}

		}

		std::cout << "" << endl;


		////////////////////////////////////////////////////////
		/////////////// Create base histogram
		////////////////////////////////////////////////////////

		// creates vector to store output
		std::vector<unsigned int> histogramData(bins);

		// creates buffer for bin calculator and the input image - buffers used in more than one Kernel
		cl::Buffer binDiv(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, pixels.size() * sizeof(unsigned int));
		// write previous two buffers to the memory buffer - buffers used in more than one Kernel
		cl::Event inIamgeTransfer;
		cl::Event dividerTransfer;
		cl::Event histOut;

		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, pixels.size() * sizeof(unsigned int), &pixels[0], NULL, &inIamgeTransfer);
		queue.enqueueWriteBuffer(binDiv, CL_TRUE, 0, sizeof(unsigned int), &binsDivider, NULL, &dividerTransfer);

		// stores the images size
		int imageSize = image_input.size();

		// Asked user to choose histogram type
		string histType;
		std::cout << "Invalid options will run default option" << endl;
		std::cout << "Please select which Histogram method you would like to run. P = Parallel(Default) S = Serial: "; // Type a number and press enter
		std::cin >> histType; // Get user input from the keyboard

		if (histType == "S" || histType == "s") {

			// Get starting timepoint
			auto start = std::chrono::high_resolution_clock::now();
			// runs serial histogram
			for (int i = 0; i < pixels.size(); i++) {


				// Calculates bin location
				const int pixel = pixels[i];
				float bin = pixel / binsDivider;
				int location = round(bin);

				// Increments bin
				histogramData[location]++;
			}
			// Get ending timepoint
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
			std::cout << "Serial histogram took " << duration.count() << "NS" << endl;
		}
		else {
			// runs parallel histogram

			cl::Event HistEvent;
			// creates bugger for the histogram
			cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
			// creates kernel and sets argumements
			cl::Kernel histogramKernel(program, "histogram");
			histogramKernel.setArg(0, dev_image_input);
			histogramKernel.setArg(1, histogramBuffer);
			histogramKernel.setArg(2, binDiv);
			// local memory argument
			//histogramKernel.setArg(3, bins * sizeof(unsigned int), NULL);
			// runs kernel
			queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(pixels.size()), cl::NDRange(1), NULL, &HistEvent);
			// reads output histogram from the buffer
			queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histogramData.size() * sizeof(unsigned int), histogramData.data(), NULL, &histOut);

			std::cout << "Kernel execution time [ns]:" << HistEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - HistEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(HistEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Image transfer time [ns]:" << inIamgeTransfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inIamgeTransfer.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "binsize transfer time [ns]:" << dividerTransfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - dividerTransfer.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output transfer time [ns]:" << histOut.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histOut.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			
		}

		ofstream histFile;
		histFile.open("Base_Histogram.csv");
		for (int i = 0; i < histogramData.size(); i++) {
			//std::cout << "Bin: " << i << " intensity: " << histogramData[i] << endl;
			histFile << i << "," << histogramData[i] << endl;
		}


		std::cout << "" << endl;


		////////////////////////////////////////////////////////
		/////////////// Create cumulative histogram
		////////////////////////////////////////////////////////

		cl::Event ScanEvent;
		cl::Event ScanInEvent;
		cl::Event ScanOutEvent;
		// creates vector to store output
		std::vector<unsigned int> CumulativeHistogramData(bins);

		// creates and writes buffer for parallel kernel histogram data
		cl::Buffer ChistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
		queue.enqueueWriteBuffer(ChistogramBuffer, CL_TRUE, 0, histogramData.size() * sizeof(unsigned int), &histogramData[0], NULL, &ScanInEvent);



		// asks user to choose scan type
		string scanType;
		std::cout << "Please select which scan method you would like to run. H = Hillis-Steele B == Blelloch(Default) S = Serial: "; // Type a number and press enter
		std::cin >> scanType; // Get user input from the keyboard
		if (scanType == "H" || scanType == "h") {


			/////////////// Runs Hillis-Steele
			std::cout << "Hillis-Steele selected" << endl;
			// creates and writes buffer for input and ouput histograms
			cl::Buffer OuthistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
			// sets up kernel for cumulative histogram histogram and passes arguments
			cl::Kernel C_histogramKernel(program, "C_histogramhs");
			C_histogramKernel.setArg(0, ChistogramBuffer);
			C_histogramKernel.setArg(1, OuthistogramBuffer);
			// runs kernel
			queue.enqueueNDRangeKernel(C_histogramKernel, cl::NullRange, cl::NDRange(histogramData.size()), cl::NDRange(bins), NULL, &ScanEvent);
			// reads output histogram from the buffer
			queue.enqueueReadBuffer(OuthistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), CumulativeHistogramData.data(), NULL, &ScanOutEvent);

			std::cout << "Kernel execution time [ns]:" << ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(ScanEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Input histogram transfer time [ns]:" << ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output histogram transfer time [ns]:" << ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		}
		else if (scanType == "S" || scanType == "s") {
			std::cout << "Serial selected" << endl;

			// Get starting timepoint
			auto start = std::chrono::high_resolution_clock::now();

			// runs serial can
			for (int i = 1; i < histogramData.size(); i++) {

				// adds current record to previous record
				histogramData[i] += histogramData[i-1];
			}
			// stores data in new vector
			CumulativeHistogramData = histogramData;

			// Get ending timepoint
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
			std::cout << "Serial scan took " << duration.count() << " NS" << endl;
		}
		else{

			// runs Blelloch
			std::cout << "Blelloch selected" << endl;
			// sets up kernel for cumulative histogram histogram and passes arguments
			cl::Kernel C_histogramKernel(program, "C_histogram");
			C_histogramKernel.setArg(0, ChistogramBuffer);
			// runs kernel
			queue.enqueueNDRangeKernel(C_histogramKernel, cl::NullRange, cl::NDRange(histogramData.size()), cl::NDRange(bins), NULL, &ScanEvent);
			// reads output histogram from the buffer
			queue.enqueueReadBuffer(ChistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), CumulativeHistogramData.data(), NULL, &ScanOutEvent);

			std::cout << "Kernel execution time [ns]:" << ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(ScanEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Input histogram transfer time [ns]:" << ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output histogram transfer time [ns]:" << ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		}
;

		ofstream CumulativeHistFile;
		CumulativeHistFile.open("Cumulative_Histogram.csv");
		for (int i = 0; i < CumulativeHistogramData.size(); i++) {
			//std::cout << "Bin: " << i << " intensity: " << CumulativeHistogramData[i] << endl;
			CumulativeHistFile << i << "," << CumulativeHistogramData[i] << endl;
		}



		std::cout << "" << endl;



		////////////////////////////////////////////////////////
		/////////////// Max and Min numbers
		////////////////////////////////////////////////////////

		// stores end number from cumulative scan as max num
		unsigned int maxNum = CumulativeHistogramData.back();
		// stors maxNum to use for finding minium
		unsigned int minNum = maxNum;



		// asks user to choose method for finding minum number
		string minType;
		std::cout << "Please select which scan method you would like to find the lowest number in the dataset. S = Serial (Default) P = Parallel: "; // Type a number and press enter
		std::cin >> minType; // Get user input from the keyboard
		if (minType == "P" || minType == "p") {

			// runs parallel reduce
			std::cout << "Parallel selected" << endl;

			// runs parallel reduce
			cl::Event MinEvent;
			cl::Event MinInEvent;
			cl::Event MinOutEvent;

			// creates and writes buffer to store output data
			cl::Buffer NhistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
			queue.enqueueWriteBuffer(NhistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), &CumulativeHistogramData[0], NULL, &MinInEvent);

			// sets up kernel for normalisation and passes arguments
			cl::Kernel reduce(program, "reduce");
			reduce.setArg(0, NhistogramBuffer);
			// creates vector to store results
			queue.enqueueNDRangeKernel(reduce, cl::NullRange, cl::NDRange(CumulativeHistogramData.size()), cl::NullRange, NULL, &MinEvent);
			std::vector<unsigned int> minStorage(bins);
			// reads results from buffer
			queue.enqueueReadBuffer(NhistogramBuffer, CL_TRUE, 0, minStorage.size() * sizeof(unsigned int), minStorage.data(), NULL, &MinOutEvent);

			std::cout << "Kernel execution time [ns]:" << MinEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - MinEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(MinEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Input min transfer time [ns]:" << MinInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - MinInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output min transfer time [ns]:" << MinOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - MinOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

			// stores stat of array as minimum number
			minNum = minStorage[0];

		
		}
		else {

			// runs serial reduce
			std::cout << "Serial selected" << endl;

			// Get starting timepoint
			auto start = std::chrono::high_resolution_clock::now();
			int i = 0;
			bool numFound = false;

			while (i < CumulativeHistogramData.size() && !numFound) {

				// Numbers are in cumulative order so the first non zero value will be the minimum;
				if (CumulativeHistogramData[i] != 0) {
					minNum = CumulativeHistogramData[i];
					numFound = true;
				}
				i++;
			}

			// Get ending timepoint
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
			std::cout << "Serial reduce took " << duration.count() << " NS" << endl;
		}

		std::cout << "" << endl;
		std::cout << "Min: " << minNum << endl;
		std::cout << "Max: " << maxNum << endl;
		// reduces size of numbers to prevent overflow
		minNum = minNum / 10;
		maxNum = maxNum / 10;



		////////////////////////////////////////////////////////
		/////////////// Histogram normalisaiton
		////////////////////////////////////////////////////////

		// creates vector to store histogram results
		std::vector<unsigned int> NormalisedHistogramData(bins);

		// asks user to choose normalisation method
		string normType;
		std::cout << "Please select which scan method you would like to use to normalise the histogram. S = Serial P = Parallel(Default): "; // Type a number and press enter
		std::cin >> normType; // Get user input from the keyboard
		if (normType == "S" || normType == "s") {

			// runs serial normalisation
			std::cout << "Serial selected" << endl;

			// Get starting timepoint
			auto start = std::chrono::high_resolution_clock::now();

			for (int i = 0; i < CumulativeHistogramData.size(); i++) {

				// reduce size of value to prevent overflow
				int currentValue = CumulativeHistogramData[i] / 10;

				// stores 0-1 normalisation
				double minScale = 0;
				double maxScale = 1;
				double normalised;

				if (i == 0 || CumulativeHistogramData[i] == 0) {
					// prevents need to calculate 0 count entries
					CumulativeHistogramData[i] = 0;
				}
				else {
					// normlises entry between 0 - 1
					normalised = minScale + (currentValue - minNum) * (maxScale - minScale) / (maxNum - minNum);
					// scales normalistion to 0 - 256
					NormalisedHistogramData[i] = normalised * 255;
				}
			}

			// Get ending timepoint
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
			std::cout << "Serial normalise took " << duration.count() << " NS" << endl;

		}
		else {

			cl::Event NormEvent;
			cl::Event NormMinEvent;
			cl::Event NormMaxEvent;
			cl::Event NormInEvent;
			cl::Event NormOutEvent;

			// runs parallel nromalosation
			std::cout << "Parallel selected" << endl;
			// creates and writes buffers for min value, max value and normalised histogram
			cl::Buffer NhistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
			cl::Buffer minNumBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
			cl::Buffer maxNumBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
			queue.enqueueWriteBuffer(NhistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), &CumulativeHistogramData[0], NULL, &NormInEvent);
			queue.enqueueWriteBuffer(minNumBuffer, CL_TRUE, 0, sizeof(unsigned int), &minNum, NULL, &NormMinEvent);
			queue.enqueueWriteBuffer(maxNumBuffer, CL_TRUE, 0, sizeof(unsigned int), &maxNum, NULL, &NormMaxEvent);
			// sets up kernel for normalisation and passes arguments
			cl::Kernel N_histogramKernel(program, "N_histogram");
			N_histogramKernel.setArg(0, NhistogramBuffer);
			N_histogramKernel.setArg(1, minNumBuffer);
			N_histogramKernel.setArg(2, maxNumBuffer);
			// runs histogram kernel
			queue.enqueueNDRangeKernel(N_histogramKernel, cl::NullRange, cl::NDRange(CumulativeHistogramData.size()), cl::NullRange, NULL, &NormEvent);
			// reads results from buffer
			queue.enqueueReadBuffer(NhistogramBuffer, CL_TRUE, 0, NormalisedHistogramData.size() * sizeof(unsigned int), NormalisedHistogramData.data(), NULL, &NormOutEvent);

			std::cout << "Kernel execution time [ns]:" << NormEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(NormEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Min transfer time [ns]:" << NormMinEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormMinEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Max transfer time [ns]:" << NormMaxEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormMaxEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Input histogram transfer time [ns]:" << NormInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output histogram transfer time [ns]:" << NormOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			
		}

		ofstream NormalisedHistFile;
		NormalisedHistFile.open("Normalised_Histogram.csv");
		for (int i = 0; i < NormalisedHistogramData.size(); i++) {
			//std::cout << "Bin: " << i << " intensity: " << CumulativeHistogramData[i] << endl;
			NormalisedHistFile << i << "," << NormalisedHistogramData[i] << endl;
		}

		std::cout << "" << endl;





		////////////////////////////////////////////////////////
		/////////////// Image equalisation
		////////////////////////////////////////////////////////
		

		// creates vector to store equalisation results
		vector<unsigned int> output_buffer(image_input.size());
		// creates vector to store equalisation results
		vector<unsigned int> temp_output_buffer(pixels.size());
		// asks user to choose normalisation method

		vector<unsigned char> convert;
		string eqType;
		std::cout << "Please select which scan method you would like to use to equalise the . S = Serial P = Parallel(Default): "; // Type a number and press enter
		std::cin >> eqType; // Get user input from the keyboard
		if (eqType == "S" || eqType == "s") {

			// Get starting timepoint
			auto start = std::chrono::high_resolution_clock::now();
			
			for (int i = 0; i < pixels.size(); i++) {
				// calculates bin location
				int in_intensity = int(pixels[i]) / binsDivider;
				int new_intensity = 0;

				// gets intensity
				temp_output_buffer[i] = NormalisedHistogramData[in_intensity];
			}

			if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {
				temp_output_buffer.insert(end(temp_output_buffer), begin(intenEnd), end(intenEnd));
				output_buffer.assign(temp_output_buffer.begin(), temp_output_buffer.end());
	
			}
			else {
				output_buffer = temp_output_buffer;

			}
			
			// Get ending timepoint
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
			std::cout << "Serial equalise took " << duration.count() << " NS" << endl;

		}
		else {

			cl::Event EqEvent;
			cl::Event EqInEvent;
			cl::Event EqOutEvent;
			// runs parallel equalisation
			std::cout << "Parallel selected" << endl;
			// creates and writes buffer for normalised histogram and output image
			cl::Buffer BPhistogramBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(unsigned int));
			cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, pixels.size() * sizeof(unsigned int)); //should be the same as input image
			queue.enqueueWriteBuffer(BPhistogramBuffer, CL_TRUE, 0, NormalisedHistogramData.size() * sizeof(unsigned int), &NormalisedHistogramData[0], NULL, &EqInEvent);



			// sets up kernel for back propogation and passes arguments
			cl::Kernel kernel = cl::Kernel(program, "equalise");
			kernel.setArg(0, dev_image_input);
			kernel.setArg(1, dev_image_output);
			kernel.setArg(2, BPhistogramBuffer);
			kernel.setArg(3, binDiv);

			// runs kernel
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(pixels.size()), cl::NDRange(1), NULL, &EqEvent);



			if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {

				// reads results from buffer
				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, temp_output_buffer.size() * sizeof(unsigned int), &temp_output_buffer.data()[0], NULL, &EqOutEvent);
				temp_output_buffer.insert(end(temp_output_buffer), begin(intenEnd), end(intenEnd));
				output_buffer.assign(temp_output_buffer.begin(), temp_output_buffer.end());



			}
			else {

				// reads results from buffer
				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size() * sizeof(unsigned int), &output_buffer.data()[0], NULL, &EqOutEvent);

			}

			std::cout << "Kernel execution time [ns]:" << EqEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - EqEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(EqEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Input image already stored in buffer" << endl;
			std::cout << "bin divider already stored in buffer" << endl;
			std::cout << "Input histogram transfer time [ns]:" << EqInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - EqInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output Image transfer time [ns]:" << EqOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - EqOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		}

		convert.assign(output_buffer.begin(), output_buffer.begin() + output_buffer.size());

		// displays input and output images to users
		CImg<unsigned char> output_image(convert.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());

		if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {
			output_image = output_image.YCbCrtoRGB();
			image_input = image_input.YCbCrtoRGB();
		}

		CImgDisplay disp_input(image_input, "input");
		CImgDisplay disp_output(output_image, "output");
		auto Mainstop = std::chrono::high_resolution_clock::now();
		auto Mainduration = std::chrono::duration_cast<std::chrono::nanoseconds>(Mainstop - Mainstart);
		std::cout << "Overall execution time: " << Mainduration.count() << " NS" << endl;

		while (!disp_input.is_closed() && !disp_output.is_closed() && !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}


	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
