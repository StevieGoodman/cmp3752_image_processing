#pragma once
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

vector<int> create_intensity_histogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cimg_library::CImg<unsigned char> from) {
	const int BIN_COUNT = 256;

	// 1. Create buffers and load image to device memory
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, from.size());
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, BIN_COUNT * 3 * sizeof(int)); // Multiply by 3 because we create one histogram for each colour channel
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, from.size(), from.data());

	// 2. Load and execute kernel
	cl::Kernel kernel = cl::Kernel(program, "create_intensity_histogram");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	cl::NDRange NDrange{ from.size() / 3 }; // Divide by number of channels to prevent repeat operations
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, NDrange, cl::NullRange);

	// 3. Retrieve output from device memory
	vector<int> histogram(BIN_COUNT * 3); // Multiply by 3 because we create one histogram for each colour channel
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, BIN_COUNT * 3 * sizeof(int), histogram.data());

	// 4. Return result
	return histogram;
}

vector<int> cumulate_histogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, vector<int> histogram) {
	const int BUFFER_SIZE = histogram.size() * sizeof(int);

	// 1. Create buffers and load image to device memory
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, BUFFER_SIZE);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, BUFFER_SIZE);
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, BUFFER_SIZE, histogram.data());

	// 2. Load and execute kernel
	cl::Kernel kernel = cl::Kernel(program, "cumulate_histogram");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	cl::NDRange NDrange{ histogram.size() / 3 }; // Divide by number of channels to prevent repeat operations
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, NDrange, cl::NullRange);

	// 3. Retrieve output from device memory
	vector<int> cumulative_histogram(histogram.size());
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, BUFFER_SIZE, cumulative_histogram.data());

	// 4. Return result
	return cumulative_histogram;
}

CImg<unsigned char> map_cumulative_histogram_to_image(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cimg_library::CImg<unsigned char> input_image, vector<int> cumulative_histogram) {
	const int HISTOGRAM_SIZE = cumulative_histogram.size() * sizeof(int);

	// 1. Create buffers and load image to device memory
	cl::Buffer input_image_buffer(context, CL_MEM_READ_ONLY, input_image.size());
	cl::Buffer input_histogram_buffer(context, CL_MEM_READ_ONLY, HISTOGRAM_SIZE);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, input_image.size());
	queue.enqueueWriteBuffer(input_image_buffer, CL_TRUE, 0, input_image.size(), input_image.data());
	queue.enqueueWriteBuffer(input_histogram_buffer, CL_TRUE, 0, HISTOGRAM_SIZE, cumulative_histogram.data());

	// 2. Load and execute kernel
	cl::Kernel kernel = cl::Kernel(program, "map_cumulative_histogram_to_image");
	kernel.setArg(0, input_image_buffer);
	kernel.setArg(1, input_histogram_buffer);
	kernel.setArg(2, output_buffer);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_image.size() / 3), cl::NullRange);

	// 3. Retrieve output from device memory
	vector<unsigned char> image_data(input_image.size());
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, input_image.size(), image_data.data());
	CImg<unsigned char> output_image(image_data.data(), input_image.width(), input_image.height(), input_image.depth(), input_image.spectrum());

	// 4. Return result
	return output_image;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.ppm";

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
		// 1. Setup OpenlCL & CImg
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		cl::Context context = GetContext(platform_id, device_id);
		cl::CommandQueue queue(context);
		cl::Program::Sources sources;
		AddSources(sources, "kernels.cl");
		cl::Program program(context, sources);
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// 2. Attempt to build kernels
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// 3. Perform histogram equalisation
		auto intensity_histogram = create_intensity_histogram(program, context, queue, image_input);
		auto cumulative_histogram = cumulate_histogram(program, context, queue, intensity_histogram);
		auto output_image = map_cumulative_histogram_to_image(program, context, queue, image_input, cumulative_histogram);

		// 4. Display output
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed()
			&& !disp_input.is_keyESC()) {
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