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
	int BIN_COUNT = from.max();

	// 1. Create buffers and load image to device memory
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, from.size());
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, BIN_COUNT * from.spectrum() * sizeof(int)); // Multiply by channels because we create one histogram for each colour channel
	cl::Buffer image_data_buffer(context, CL_MEM_READ_WRITE, sizeof(int) * 2);
	cl::Event input_event;
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, from.size(), from.data(), NULL, &input_event);
	vector<int> image_data{ from.max(), from.spectrum() };
	queue.enqueueWriteBuffer(image_data_buffer, CL_TRUE, 0, sizeof(int) * image_data.size(), image_data.data());

	// 2. Load and execute kernel
	cl::Kernel kernel = cl::Kernel(program, "create_intensity_histogram");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	kernel.setArg(2, image_data_buffer);
	cl::NDRange NDrange{ from.size() / from.spectrum() }; // Divide by number of channels to prevent repeat operations
	cl::Event kernel_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, NDrange, cl::NullRange, NULL, &kernel_event);

	// 3. Retrieve output from device memory
	vector<int> histogram(BIN_COUNT * from.spectrum()); // Multiply by 3 because we create one histogram for each colour channel
	cl::Event output_event;
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, BIN_COUNT * from.spectrum() * sizeof(int), histogram.data(), NULL, &output_event);

	// 4. Return result
	cout << "[ CREATE INTENSITY HISTOGRAM ]" << endl;
	cout << "Load image buffer: " << GetFullProfilingInfo(input_event, PROF_US) << endl;
	cout << "Generate intensity histogram: " << GetFullProfilingInfo(kernel_event, PROF_US) << endl;
	cout << "Retrieve histogram : " << GetFullProfilingInfo(output_event, PROF_US) << endl;

	return histogram;
}

vector<int> cumulate_histogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, vector<int> histogram, int channels) {
	const int BUFFER_SIZE = histogram.size() * sizeof(int);

	// 1. Create buffers and load image to device memory
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, BUFFER_SIZE);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, BUFFER_SIZE);
	cl::Buffer channel_count_buffer(context, CL_MEM_READ_ONLY, sizeof(int));
	cl::Event input_event;
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, BUFFER_SIZE, histogram.data(), NULL, &input_event);
	std::vector<int> channel_count_vector{ channels };
	queue.enqueueWriteBuffer(channel_count_buffer, CL_TRUE, 0, sizeof(int), channel_count_vector.data());

	// 2. Load and execute kernel
	cl::Kernel kernel = cl::Kernel(program, "cumulate_histogram");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	kernel.setArg(2, channel_count_buffer);
	cl::NDRange NDrange{ histogram.size() / channels }; // Divide by number of channels to prevent repeat operations
	cl::Event kernel_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, NDrange, cl::NullRange, NULL, &kernel_event);

	// 3. Retrieve output from device memory
	vector<int> cumulative_histogram(histogram.size());
	cl::Event output_event;
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, BUFFER_SIZE, cumulative_histogram.data(), NULL, &output_event);

	// 4. Return result
	cout << "[ CUMULATE HISTOGRAM ]" << endl;
	cout << "Load histogram buffer: " << GetFullProfilingInfo(input_event, PROF_US) << endl;
	cout << "Generate cumulative histogram: " << GetFullProfilingInfo(kernel_event, PROF_US) << endl;
	cout << "Retrieve cumulative histogram : " << GetFullProfilingInfo(output_event, PROF_US) << endl;
	return cumulative_histogram;
}

CImg<unsigned char> map_cumulative_histogram_to_image(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, cimg_library::CImg<unsigned char> input_image, vector<int> cumulative_histogram) {
	const int HISTOGRAM_SIZE = cumulative_histogram.size() * sizeof(int);

	// 1. Create buffers and load image to device memory
	cl::Buffer input_image_buffer(context, CL_MEM_READ_ONLY, input_image.size());
	cl::Buffer input_histogram_buffer(context, CL_MEM_READ_ONLY, HISTOGRAM_SIZE);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, input_image.size());
	cl::Buffer image_data_buffer(context, CL_MEM_READ_ONLY, sizeof(int) * 2);
	cl::Event input_image_event;
	cl::Event input_histogram_event;
	queue.enqueueWriteBuffer(input_image_buffer, CL_TRUE, 0, input_image.size(), input_image.data(), NULL, &input_image_event);
	queue.enqueueWriteBuffer(input_histogram_buffer, CL_TRUE, 0, HISTOGRAM_SIZE, cumulative_histogram.data(), NULL, &input_histogram_event);
	vector<int> image_data{ input_image.max(), input_image.spectrum() };
	queue.enqueueWriteBuffer(image_data_buffer, CL_TRUE, 0, sizeof(int) * image_data.size(), image_data.data());

	// 2. Load and execute kernel
	cl::Kernel kernel = cl::Kernel(program, "map_cumulative_histogram_to_image");
	kernel.setArg(0, input_image_buffer);
	kernel.setArg(1, input_histogram_buffer);
	kernel.setArg(2, output_buffer);
	kernel.setArg(3, image_data_buffer);
	cl::Event kernel_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_image.size() / input_image.spectrum()), cl::NullRange, NULL, &kernel_event);

	// 3. Retrieve output from device memory
	vector<unsigned char> image(input_image.size());
	cl::Event output_event;
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, input_image.size(), image.data(), NULL, &output_event);
	CImg<unsigned char> output_image(image.data(), input_image.width(), input_image.height(), input_image.depth(), input_image.spectrum());

	// 4. Return result
	cout << "[ MAP CUMULATIVE HISTOGRAM TO IMAGE ]" << endl;
	cout << "Load image buffer: " << GetFullProfilingInfo(input_image_event, PROF_US) << endl;
	cout << "Load histogram buffer: " << GetFullProfilingInfo(input_histogram_event, PROF_US) << endl;
	cout << "Generate modified image: " << GetFullProfilingInfo(kernel_event, PROF_US) << endl;
	cout << "Retrieve modified image : " << GetFullProfilingInfo(output_event, PROF_US) << endl;
	return output_image;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "mdr16-gs.pgm"; // Valid: test.pgm, test.ppm, test_large.pgm, test_large.ppm, mdr-16.ppm, mdr16-gs.pgm. NOTE: 16-bit images seem to have issues with high-intensity values, but I'm not quite sure why.

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
		CImg<unsigned short> image_query(image_filename.c_str());
		CImg<unsigned char> image_input(image_filename.c_str());
		bool bit16 = image_query.max() > 255; // Perform 16-to-8 bit conversion if necessary
		if (bit16)
			image_input = image_query.normalize(0, 255);

		CImgDisplay disp_input(image_input, "input");

		cl::Context context = GetContext(platform_id, device_id);
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
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
		auto cumulative_histogram = cumulate_histogram(program, context, queue, intensity_histogram, image_input.spectrum());
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