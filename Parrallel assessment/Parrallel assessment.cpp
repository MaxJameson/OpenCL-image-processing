#include <iostream>
#include <vector>

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
	//Part 1 - handle command line options such as device selection, verbosity, etc.
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

		bool colour = false;

		CImg<unsigned char> image_input(image_filename.c_str());


		std::vector<unsigned char> pixels;
		std::vector<unsigned char> ColourEnd;

		if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {
			image_input = image_input.RGBtoYCbCr();
			pixels.assign(image_input.begin(), image_input.begin() + (image_input.size() / 3));
			ColourEnd.assign(image_input.begin() + (image_input.size() / 3) + 1, image_input.end());
			std::cout << &image_input << endl;
			
	
		}
		else {
			pixels.assign(image_input.begin() + 0, image_input.begin() + image_input.size());
		}


		

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
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

		bool binCheck = false;
		unsigned int bins;
		unsigned int bits = 256;
		unsigned int binsDivider;
		while (!binCheck) {
			cout << "Please enter the number of bins you would like between the range of 32 - 256: "; // Type a number and press enter
			cin >> bins; // Get user input from the keyboard

			if (bins < 32 || bins > 256) {
				std::cout << "Invalid input " << endl;
				cin.clear();
				cin.ignore(1, '\n');
			}
			else {
				binCheck = true;
				if (bins = bits) {
					binsDivider = 1;
				}
				else {
					binsDivider = bits / bins;
				}
			}

		}


		/////////////// Create base histogram - histogram kernel

		// creates and writes buffers for input image and histogram data
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
		cl::Buffer binDiv(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
		cl::Buffer numBins(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, pixels.size(), &pixels[0]);
		queue.enqueueWriteBuffer(binDiv, CL_TRUE, 0, sizeof(unsigned int), &binsDivider, NULL);
		queue.enqueueWriteBuffer(numBins, CL_TRUE, 0, sizeof(unsigned int), &bins, NULL);

		// creates kernel and sets argumements
		cl::Kernel histogramKernel(program, "histogram");
		histogramKernel.setArg(0, dev_image_input);
		histogramKernel.setArg(1, histogramBuffer);
		histogramKernel.setArg(2, binDiv);
		histogramKernel.setArg(3, numBins);
		// local memory argument
		//histogramKernel.setArg(4, bins * sizeof(unsigned int), NULL);


		// runs kernel
		queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		
		// creates vector to store output
		std::vector<unsigned int> histogramData(bins);
		// reads output histogram from the buffer
		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0,histogramData.size() * sizeof(unsigned int),histogramData.data());

		for (int i = 0; i < histogramData.size(); i++) {
			std::cout << "Bin: " << i << " intensity: " << histogramData[i] << endl;
		}

		/////////////// Create cumulative histogram - scan Blelloch
		string scanType;
		cout << "Please select which scan method you would like to run. H = Hillis-Steele B == Blelloch: "; // Type a number and press enter
		cin >> scanType; // Get user input from the keyboard
		if (scanType == "H" || scanType == "h") {
			std::cout << "Hillis-Steele selected" << endl;
		}
		else{
			if(scanType != "B" || scanType != "b"){
				std::cout << "Invalid selection, Default = Blelloch" << endl;
			}
			std::cout << "Blelloch selected" << endl;
		}


		// creates and writes buffer for histogram data
		cl::Buffer ChistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
		queue.enqueueWriteBuffer(ChistogramBuffer, CL_TRUE, 0, histogramData.size() * sizeof(unsigned int), &histogramData[0], NULL);

		// sets up kernel for cumulative histogram histogram and passes arguments
		cl::Kernel C_histogramKernel(program, "C_histogram");
		C_histogramKernel.setArg(0, ChistogramBuffer);

		// runs kernel
		queue.enqueueNDRangeKernel(C_histogramKernel, cl::NullRange, cl::NDRange(histogramData.size()), cl::NullRange);

		// creates vector to store output
		std::vector<unsigned int> CumulativeHistogramData(bins);
		// reads output histogram from the buffer
		queue.enqueueReadBuffer(ChistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), CumulativeHistogramData.data());

		for (int i = 0; i < CumulativeHistogramData.size(); i++) {
			std::cout << "Bin: " << i << " intensity: " << CumulativeHistogramData[i] << endl;
		}


		/////////////// Create normalised histogram - Map
		unsigned int maxNum = *max_element(CumulativeHistogramData.begin(), CumulativeHistogramData.end());
		unsigned int minNum = maxNum;
		for (int i = 0; i < CumulativeHistogramData.size(); i++) {
			if (CumulativeHistogramData[i] != 0 && minNum > CumulativeHistogramData[i]) {
				minNum = CumulativeHistogramData[i];
			}
		}
		std::cout << minNum << endl;
		std::cout << maxNum << endl;

		// stores the smallest and largest intensity values in the picture - !!!!!!!!!!!!!!!!!!!!!! May use reduce to find
		//unsigned int minNum = *min_element(CumulativeHistogramData.begin() + 1, CumulativeHistogramData.end());
		minNum = minNum / 10;
		maxNum = maxNum / 10;

		// creates and writes buffers for min value, max value and normalised histogram
		cl::Buffer minNumBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
		cl::Buffer maxNumBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
		cl::Buffer NhistogramBuffer(context, CL_MEM_READ_WRITE, bins * sizeof(unsigned int));
		queue.enqueueWriteBuffer(minNumBuffer, CL_TRUE, 0, sizeof(unsigned int), &minNum, NULL);
		queue.enqueueWriteBuffer(maxNumBuffer, CL_TRUE, 0, sizeof(unsigned int), &maxNum, NULL);
		queue.enqueueWriteBuffer(NhistogramBuffer, CL_TRUE, 0, CumulativeHistogramData.size() * sizeof(unsigned int), &CumulativeHistogramData[0], NULL);

		// sets up kernel for normalisation and passes arguments
		cl::Kernel N_histogramKernel(program, "N_histogram");
		N_histogramKernel.setArg(0, NhistogramBuffer);
		N_histogramKernel.setArg(1, minNumBuffer);
		N_histogramKernel.setArg(2, maxNumBuffer);

		// runs histogram kernel
		queue.enqueueNDRangeKernel(N_histogramKernel, cl::NullRange, cl::NDRange(CumulativeHistogramData.size()), cl::NullRange);

		// creates vector to store histogram results
		std::vector<unsigned int> NormalisedHistogramData(bins);
		// reads results from buffer
		queue.enqueueReadBuffer(NhistogramBuffer, CL_TRUE, 0, NormalisedHistogramData.size() * sizeof(unsigned int), NormalisedHistogramData.data());

		//for (int i = 0; i < NormalisedHistogramData.size(); i++) {
			//std::cout << "Bin: "<< i << " intensity: " << NormalisedHistogramData[i] << endl;
		//}


		/////////////// Equalise Image - Map

		// creates and writes buffer for normalised histogram and output image
		cl::Buffer BPhistogramBuffer(context, CL_MEM_READ_ONLY, bins * sizeof(unsigned int));
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, pixels.size()); //should be the same as input image
		queue.enqueueWriteBuffer(BPhistogramBuffer, CL_TRUE, 0, NormalisedHistogramData.size() * sizeof(unsigned int), &NormalisedHistogramData[0], NULL);

		// sets up kernel for back propogation and passes arguments
		cl::Kernel kernel = cl::Kernel(program, "equalise");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_image_output);
		kernel.setArg(2, BPhistogramBuffer);
		kernel.setArg(3, binDiv);

		// runs kernel
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(pixels.size()), cl::NullRange);

		// creates vector to store equalisation results
		vector<unsigned char> output_buffer(image_input.size());

		if (image_filename.substr(image_filename.find_last_of(".") + 1) == "ppm") {

			// creates vector to store equalisation results
			vector<unsigned char> output_Colour_buffer(pixels.size());
			// reads results from buffer
			queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_Colour_buffer.size(), &output_Colour_buffer.data()[0]);
			output_Colour_buffer.insert(end(output_Colour_buffer), begin(ColourEnd), end(ColourEnd));
			output_buffer.assign(output_Colour_buffer.begin(), output_Colour_buffer.end());

			// displays input and output images to users
			CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
			output_image = output_image.YCbCrtoRGB();
			image_input = image_input.YCbCrtoRGB();
			CImgDisplay disp_input(image_input, "input");
			CImgDisplay disp_output(output_image, "output");

			while (!disp_input.is_closed() && !disp_output.is_closed()
				&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
				disp_input.wait(1);
				disp_output.wait(1);
			}

		}
		else {

			// reads results from buffer
			queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

			// displays input and output images to users
			CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
			CImgDisplay disp_input(image_input, "input");
			CImgDisplay disp_output(output_image, "output");

			while (!disp_input.is_closed() && !disp_output.is_closed()
				&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
				disp_input.wait(1);
				disp_output.wait(1);
			}
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
