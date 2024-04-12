// 1. Create histograms containing image intensities for each of the 3 colour channels.
// The output array is comprised of 3 contiguous intensity histograms
kernel void create_intensity_histogram(global const uchar* input, global int* output, global int* channel_count) {
	const int GID = get_global_id(0);
	const int PIXEL_COUNT = get_global_size(0);
	for (int channel = 0; channel < channel_count[0]; channel++) {
		int intensity = input[GID + (PIXEL_COUNT * channel)];
		atomic_inc(&output[intensity + (256 * channel)]);
	}
}

// 2. Creates a cumulative histogram from a non-cumulative histogram by employing a Hillis-Steele scan
// NOTE: Is "cumulate" even a verb? Too bad!
kernel void cumulate_histogram(global int* input, global int* output, global int* channel_count) {
	int GID = get_global_id(0);
	int BIN_COUNT = get_global_size(0);
	global int* swap_buffer;
	for (int stride = 1; stride <= BIN_COUNT; stride *= 2) {
		for (int channel = 0; channel < channel_count[0]; channel++) {
			output[GID + (BIN_COUNT * channel)] = input[GID + (BIN_COUNT * channel)];
			if (GID >= stride)
				output[GID + (BIN_COUNT * channel)] += input[GID + (BIN_COUNT * channel) - stride];
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		swap_buffer = input;
		input = output;
		output = swap_buffer;
	}
}

// 3. Normalize cumulative histogram
// 4. Map image pixels to CDF intensities
kernel void map_cumulative_histogram_to_image(global const uchar* input_image, global const int* histogram, global uchar* output_image, global int* channel_count) {
	const int GID = get_global_id(0);
	const int PIXEL_COUNT = get_global_size(0);
	for (int channel = 0; channel < channel_count[0]; channel++) {
		int value = (int)(((float)histogram[input_image[GID + (PIXEL_COUNT * channel)]] / (float)PIXEL_COUNT) * 255); // Prepare for... unforeseen consequencessss...
		output_image[GID + (PIXEL_COUNT * channel)] = value;
	}
	//int red_value = (int)(((float)histogram[input_image[GID]] / (float)PIXEL_COUNT) * 255);
	//int green_value = (int)(((float)histogram[input_image[GID+PIXEL_COUNT] + 256] / (float)PIXEL_COUNT) * 255);
	//int blue_value = (int)(((float)histogram[input_image[GID + PIXEL_COUNT*2] + 512] / (float)PIXEL_COUNT) * 255);
	//output_image[GID] = red_value;
	//output_image[GID + PIXEL_COUNT] = green_value;
	//output_image[GID + (2 * PIXEL_COUNT)] = blue_value;
}