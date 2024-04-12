// 1. Create histograms containing image intensities for each of the 3 colour channels.
// The output array is comprised of 3 contiguous intensity histograms
kernel void create_intensity_histogram(global const uchar* input, global int* output, global int* image_data) {
	const int BIT_DEPTH = image_data[0];
	const int CHANNEL_COUNT = image_data[1];
	//printf("BIT DEPTH: %d, CHANNEL_COUNT: %d\n", BIT_DEPTH, CHANNEL_COUNT);
	const int GID = get_global_id(0);
	const int PIXEL_COUNT = get_global_size(0);
	for (int channel = 0; channel < CHANNEL_COUNT; channel++) {
		int intensity = input[GID + (PIXEL_COUNT * channel)];
		atomic_inc(&output[intensity + (BIT_DEPTH * channel)]);
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
kernel void map_cumulative_histogram_to_image(global const uchar* input_image, global const int* histogram, global uchar* output_image, global int* image_data) {
	const int BIT_DEPTH = image_data[0];
	const int CHANNEL_COUNT = image_data[1];
	const int GID = get_global_id(0);
	const int PIXEL_COUNT = get_global_size(0);
	for (int channel = 0; channel < CHANNEL_COUNT; channel++) {
		int value = (int)(((float)histogram[input_image[GID + (PIXEL_COUNT * channel)]] / (float)PIXEL_COUNT) * (BIT_DEPTH-1)); // Prepare for... unforeseen consequencessss...
		output_image[GID + (PIXEL_COUNT * channel)] = value;
	}
}