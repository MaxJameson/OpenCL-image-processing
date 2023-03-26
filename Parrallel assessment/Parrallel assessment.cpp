#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

int gcd(int a, int b)
{
	int result = min(a, b); // Find Minimum of a and b
	while (result > 0) {
		if (a % result == 0 && b % result == 0) {
			break;
		}
		result--;
	}
	return result; // return gcd of a and b
}



void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {

	// starts timer for overall execution time
	auto Mainstart = std::chrono::high_resolution_clock::now();

	// sets default inputs
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	// stores input argumets
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

		// sets the openCL context
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;


		// sets up command queue
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


		// converts input file to a Cimg
		CImg<unsigned char> image_input(image_filename.c_str());


		// stores each pixel in an image
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

			// creates vector of intensity values and vector of chroma red + blue
			pixels.assign(image_input.begin(), image_input.begin() + image_input.size());

		}
			 
		// stores bool to check if the value of bins is valid
		bool binCheck = false;

		// stores number of bins and max intesity value per pixel
		unsigned int bins;
		unsigned int bits = 256;

		// stores valid to calculate which bin a pixel belongs too
		unsigned int binsDivider;

		// loops until a valid bin is input
		while (!binCheck) {

			// takes input for bin
			std::cout << "Please enter the number of bins you would like between the range of 32 - " << bits << ": "; // Type a number and press enter
			std::cin >> bins; // Get user input from the keyboard

			// checsk if input is valid
			if (bins < 32 || bins > bits) {
				std::cout << "Invalid input " << endl;
				std::cin.clear();
				std::cin.ignore(1, '\n');
			}
			else {
				binCheck = true;

				// calculates the number used to define which bin and intensity belongs too
				if (bins == bits) {
					binsDivider = 1;
				}
				else {
					binsDivider = bits / bins;
				}
			}

		}



		////////////////////////////////////////////////////////
		/////////////// Create base histogram
		////////////////////////////////////////////////////////

		// creates vector to store output
		std::vector<unsigned int> histogramData(bins);

		// creates events to track runtime
		cl::Event inIamgeTransfer;
		cl::Event dividerTransfer;
		cl::Event histOut;

		// creates buffer for bin calculator and the input image - buffers used in more than one Kernel
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, pixels.size() * sizeof(unsigned int));
		cl::Buffer binDiv(context, CL_MEM_READ_ONLY, sizeof(unsigned int));

		// write previous two buffers to the memory buffer - buffers used in more than one Kernel
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, pixels.size() * sizeof(unsigned int), &pixels[0], NULL, &inIamgeTransfer);
		queue.enqueueWriteBuffer(binDiv, CL_TRUE, 0, sizeof(unsigned int), &binsDivider, NULL, &dividerTransfer);

		// stores the images size
		int imageSize = image_input.size();

		// Asked user to choose histogram type
		string histType;
		std::cout << "Invalid options will run default option" << endl;
		std::cout << "Please select which Histogram method you would like to run. P = Parallel(Default) S = Serial: "; // Type a number and press enter
		// Get user input from the keyboard
		std::cin >> histType; 

		if (histType == "S" || histType == "s") {

			std::cout << "Serial selected: " << endl;
			// starts timer for histogram creator
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
			// ends timer
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

			// outputs execution time
			std::cout << "Serial histogram took " << duration.count() << "NS" << endl;
		}
		else {

			// runs parallel histogram
			std::cout << "Parallel selected: " << endl;

			// create event to track runtime
			cl::Event HistEvent;

			// creates bugger for the histogram
			cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));

			// creates kernel and sets argumements
			cl::Kernel histogramKernel(program, "histogram");
			cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
			int hcf = gcd(pixels.size(), histogramKernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device));


			histogramKernel.setArg(0, dev_image_input);
			histogramKernel.setArg(1, histogramBuffer);
			histogramKernel.setArg(2, binDiv);
			//histogramKernel.setArg(3, cl::Local(histogramData.size() * sizeof(unsigned int)));



			// runs kernel
			queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(pixels.size()), cl::NDRange(hcf), NULL, &HistEvent);
			// reads output histogram from the buffer
			queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histogramData.size() * sizeof(unsigned int), histogramData.data(), NULL, &histOut);

			// outputs histogram runtime along with memeory transfer time
			std::cout << "Kernel execution time [ns]:" << HistEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - HistEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(HistEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Image transfer time [ns]:" << inIamgeTransfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - inIamgeTransfer.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "binsize transfer time [ns]:" << dividerTransfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - dividerTransfer.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output transfer time [ns]:" << histOut.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histOut.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			
		}

		// wrties histogram to a csv file
		ofstream histFile;
		histFile.open("Base_Histogram.csv");
		for (int i = 0; i < histogramData.size(); i++) {
			histFile << i << "," << histogramData[i] << endl;
		}


		std::cout << "" << endl;


		////////////////////////////////////////////////////////
		/////////////// Create cumulative histogram
		////////////////////////////////////////////////////////

		// creates events to track runtime
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
		string LorG;
		std::cout << "Please select which scan method you would like to run. H = Hillis-Steele B == Blelloch(Default) S = Serial: "; // Type a number and press enter
		std::cin >> scanType; // Get user input from the keyboard
		if (scanType == "H" || scanType == "h") {

			/////////////// Runs Hillis-Steele
			std::cout << "Hillis-Steele selected" << endl;
			// creates and writes buffer for input and ouput histograms
			cl::Buffer OuthistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
			// sets up kernel for cumulative histogram histogram and passes arguments

			// asks user to choose between a local and global scan
			std::cout << "Would you like to run in L = local or G = global(Default): "; // Type a number and press enter
			std::cin >> LorG; // Get user input from the keyboard

			if (LorG == "L" || LorG == "l") {

				std::cout << "Local selected" << endl;

				// runs kernel for local Hillis-steele scan
				cl::Kernel C_histogramKernel(program, "C_histogramhs_Local");
				C_histogramKernel.setArg(0, ChistogramBuffer);
				C_histogramKernel.setArg(1, OuthistogramBuffer);
				C_histogramKernel.setArg(2, cl::Local(histogramData.size() * sizeof(unsigned int)));
				C_histogramKernel.setArg(3, cl::Local(histogramData.size() * sizeof(unsigned int)));
				queue.enqueueNDRangeKernel(C_histogramKernel, cl::NullRange, cl::NDRange(histogramData.size()), cl::NDRange(bins), NULL, &ScanEvent);
			}
			else {

				std::cout << "Global selected" << endl;
				// runs kernel for global Hillis-steele scan
				cl::Kernel C_histogramKernel(program, "C_histogramhs");
				C_histogramKernel.setArg(0, ChistogramBuffer);
				C_histogramKernel.setArg(1, OuthistogramBuffer);
				queue.enqueueNDRangeKernel(C_histogramKernel, cl::NullRange, cl::NDRange(histogramData.size()), cl::NDRange(bins), NULL, &ScanEvent);
			}



			// reads output histogram from the buffer
			queue.enqueueReadBuffer(OuthistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), CumulativeHistogramData.data(), NULL, &ScanOutEvent);

			// outputs histogram runtime along with memeory transfer time
			std::cout << "Kernel execution time [ns]:" << ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(ScanEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Input histogram transfer time [ns]:" << ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output histogram transfer time [ns]:" << ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		}
		else if (scanType == "S" || scanType == "s") {
			std::cout << "Serial selected" << endl;

			// runs serial scan

			// starts timer for serial scan
			auto start = std::chrono::high_resolution_clock::now();

			// runs serial can
			for (int i = 1; i < histogramData.size(); i++) {

				// adds current record to previous record
				histogramData[i] += histogramData[i-1];
			}
			// stores data in new vector
			CumulativeHistogramData = histogramData;

			// ends timer
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
			std::cout << "Serial scan took " << duration.count() << " NS" << endl;
		}
		else{

			// runs Blelloch
			std::cout << "Blelloch selected" << endl;

			// asks user to choose between a local and global scan
			std::cout << "Would you like to run in L = local or G = global(Default): ";
			std::cin >> LorG;

			if (LorG == "L" || LorG == "l") {

				// runs kernel for local Blelloch scan
				cl::Kernel C_histogramKernel(program, "C_histogram_Local");
				C_histogramKernel.setArg(0, ChistogramBuffer);
				C_histogramKernel.setArg(1, cl::Local(histogramData.size() * sizeof(unsigned int)));
				queue.enqueueNDRangeKernel(C_histogramKernel, cl::NullRange, cl::NDRange(histogramData.size()), cl::NDRange(bins), NULL, &ScanEvent);
			}
			else {

				// runs kernel for global Blelloch scan
				cl::Kernel C_histogramKernel(program, "C_histogram");
				C_histogramKernel.setArg(0, ChistogramBuffer);
				queue.enqueueNDRangeKernel(C_histogramKernel, cl::NullRange, cl::NDRange(histogramData.size()), cl::NDRange(bins), NULL, &ScanEvent);
			}

			// reads output histogram from the buffer
			queue.enqueueReadBuffer(ChistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), CumulativeHistogramData.data(), NULL, &ScanOutEvent);

			// outputs histogram runtime along with memeory transfer time
			std::cout << "Kernel execution time [ns]:" << ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(ScanEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Input histogram transfer time [ns]:" << ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output histogram transfer time [ns]:" << ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - ScanOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		}
;
		
		// outputs histogram to a csv file
		ofstream CumulativeHistFile;
		CumulativeHistFile.open("Cumulative_Histogram.csv");
		for (int i = 0; i < CumulativeHistogramData.size(); i++) {
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
		std::cout << "Please select which scan method you would like to find the lowest number in the dataset. S = Serial (Default) P = Parallel: ";
		std::cin >> minType;
		if (minType == "P" || minType == "p") {

			// runs parallel reduce
			std::cout << "Parallel selected" << endl;

			// creates events to time memory transfer
			cl::Event MinEvent;
			cl::Event MinInEvent;
			cl::Event MinOutEvent;

			// creates and writes buffer to store output data
			cl::Buffer NhistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
			queue.enqueueWriteBuffer(NhistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), &CumulativeHistogramData[0], NULL, &MinInEvent);

			// runs kernal to find the minimun non zero number in a dataset
			cl::Kernel reduce(program, "reduce");
			reduce.setArg(0, NhistogramBuffer);
			queue.enqueueNDRangeKernel(reduce, cl::NullRange, cl::NDRange(CumulativeHistogramData.size()), cl::NDRange(bins), NULL, &MinEvent);

			// reads results from buffer
			std::vector<unsigned int> minStorage(bins);
			queue.enqueueReadBuffer(NhistogramBuffer, CL_TRUE, 0, minStorage.size() * sizeof(unsigned int), minStorage.data(), NULL, &MinOutEvent);

			// outputs histogram runtime along with memeory transfer time
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

			// starts time for serial reduce
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

		// reduces size of numbers to prevent overflow when normalising larger images
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

			// starts timer for serial normalistion
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
					NormalisedHistogramData[i] = normalised * bits;
				}
			}

			// Get ending timepoint
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
			std::cout << "Serial normalise took " << duration.count() << " NS" << endl;

		}
		else {

			// creates events to track normalisation kernel
			cl::Event NormEvent;
			cl::Event NormMinEvent;
			cl::Event NormMaxEvent;
			cl::Event NormInEvent;
			cl::Event NormOutEvent;

			// runs parallel normalisation
			std::cout << "Parallel selected" << endl;
			// creates and writes buffers for min value, max value and normalised histogram
			cl::Buffer NhistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
			cl::Buffer minNumBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
			cl::Buffer maxNumBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
			cl::Buffer bitsBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
			queue.enqueueWriteBuffer(NhistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), &CumulativeHistogramData[0], NULL, &NormInEvent);
			queue.enqueueWriteBuffer(minNumBuffer, CL_TRUE, 0, sizeof(unsigned int), &minNum, NULL, &NormMinEvent);
			queue.enqueueWriteBuffer(maxNumBuffer, CL_TRUE, 0, sizeof(unsigned int), &maxNum, NULL, &NormMaxEvent);
			queue.enqueueWriteBuffer(bitsBuffer, CL_TRUE, 0, sizeof(unsigned int), &bits);

			// runs normalistaion kernel
			cl::Kernel N_histogramKernel(program, "N_histogram");
			N_histogramKernel.setArg(0, NhistogramBuffer);
			N_histogramKernel.setArg(1, minNumBuffer);
			N_histogramKernel.setArg(2, maxNumBuffer);
			N_histogramKernel.setArg(3, bitsBuffer);
			queue.enqueueNDRangeKernel(N_histogramKernel, cl::NullRange, cl::NDRange(CumulativeHistogramData.size()), cl::NullRange, NULL, &NormEvent);
			// reads results from buffer
			queue.enqueueReadBuffer(NhistogramBuffer, CL_TRUE, 0, NormalisedHistogramData.size() * sizeof(unsigned int), NormalisedHistogramData.data(), NULL, &NormOutEvent);

			// outputs histogram runtime along with memeory transfer time
			std::cout << "Kernel execution time [ns]:" << NormEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(NormEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Min transfer time [ns]:" << NormMinEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormMinEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Max transfer time [ns]:" << NormMaxEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormMaxEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Input histogram transfer time [ns]:" << NormInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output histogram transfer time [ns]:" << NormOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - NormOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			
		}

		// outputs normalised histogram to a .csv file
		ofstream NormalisedHistFile;
		NormalisedHistFile.open("Normalised_Histogram.csv");
		for (int i = 0; i < NormalisedHistogramData.size(); i++) {
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
		// stores converted image
		vector<unsigned char> convert(image_input.size());

		// asks user to select which equlisation they want to use
		string eqType;
		std::cout << "Please select which scan method you would like to use to equalise the . S = Serial P = Parallel(Default): "; 
		std::cin >> eqType;
		if (eqType == "S" || eqType == "s") {

			// starts timer to track serial equalise 
			auto start = std::chrono::high_resolution_clock::now();
			
			// maps image to new intensities
			for (int i = 0; i < pixels.size(); i++) {
				// calculates bin location
				int in_intensity = int(pixels[i]) / binsDivider;
				int new_intensity = 0;

				// gets intensity
				temp_output_buffer[i] = NormalisedHistogramData[in_intensity];
			}

			// checks if the image is colour or greyscale
			if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {

				// adds the intensity pixels back to the main image
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

			// runs parallel equalisation
			std::cout << "Parallel selected" << endl;

			// creates events to track equalistion
			cl::Event EqEvent;
			cl::Event EqInEvent;
			cl::Event EqOutEvent;

			// creates and writes buffer for normalised histogram and output image
			cl::Buffer BPhistogramBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(unsigned int));
			cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, pixels.size() * sizeof(unsigned int)); //should be the same as input image
			queue.enqueueWriteBuffer(BPhistogramBuffer, CL_TRUE, 0, NormalisedHistogramData.size() * sizeof(unsigned int), &NormalisedHistogramData[0], NULL, &EqInEvent);



			// runs kernel for requalisation
			cl::Kernel equal(program, "equalise");
			equal.setArg(0, dev_image_input);
			equal.setArg(1, dev_image_output);
			equal.setArg(2, BPhistogramBuffer);
			equal.setArg(3, binDiv);
			queue.enqueueNDRangeKernel(equal, cl::NullRange, cl::NDRange(pixels.size()), cl::NDRange(1), NULL, &EqEvent);


			// checks if the image is colour or greyscale
			if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {

				// reads buffer and adds the intensity pixels back to the main image
				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, temp_output_buffer.size() * sizeof(unsigned int), &temp_output_buffer.data()[0], NULL, &EqOutEvent);
				temp_output_buffer.insert(end(temp_output_buffer), begin(intenEnd), end(intenEnd));
				output_buffer.assign(temp_output_buffer.begin(), temp_output_buffer.end());



			}
			else {

				// reads results from buffer
				queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size() * sizeof(unsigned int), &output_buffer.data()[0], NULL, &EqOutEvent);

			}

			// outputs histogram runtime along with memeory transfer time
			std::cout << "Kernel execution time [ns]:" << EqEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - EqEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << GetFullProfilingInfo(EqEvent, ProfilingResolution::PROF_NS) << std::endl;
			std::cout << "Input image already stored in buffer" << endl;
			std::cout << "bin divider already stored in buffer" << endl;
			std::cout << "Input histogram transfer time [ns]:" << EqInEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - EqInEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
			std::cout << "Output Image transfer time [ns]:" << EqOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - EqOutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		}


		// converts vector of ints to vector of char
		convert.assign(output_buffer.begin(), output_buffer.begin() + output_buffer.size());

		// adds equalised pixels to a CImg
		CImg<unsigned char> output_image(convert.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());

		// reconverts colour space of colour image
		if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {
			output_image = output_image.YCbCrtoRGB();
			image_input = image_input.YCbCrtoRGB();
		}

		// displays original and equalised images
		CImgDisplay disp_input(image_input, "input");
		CImgDisplay disp_output(output_image, "output");

		// stops and displays overall program timer
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

