// 1. Create histogram containing image intensities.
kernel void create_intensity_histogram(global const uchar* input, global int* output) {
	int gid = get_global_id(0);
	int intensity = input[gid];
	atomic_inc(&output[intensity]);
}

// 2. Creates a cumulative histogram from a non-cumulative histogram by employing a Hillis-Steele scan
// NOTE: Is "cumultate" even a verb? Oh well!
kernel void cumulate_histogram(global int* input, global int* output) {
	int gid = get_global_id(0);
	int gsize = get_global_size(0);
	global int* swap_buffer;
	for (int stride = 1; stride <= gsize; stride *= 2) {
		output[gid] = input[gid];
		if (gid >= stride)
			output[gid] += input[gid - stride];

		barrier(CLK_GLOBAL_MEM_FENCE);

		swap_buffer = input;
		input = output;
		output = swap_buffer;
	}
}

// 3. Normalize cumulative histogram
// 4. Map image pixels to CDF intensities
kernel void map_cumulative_histogram_to_image(global const uchar* input_image, global const int* histogram, global uchar* output_image) {
	int gid = get_global_id(0);
	const int PIXEL_COUNT = get_global_size(0);
	output_image[gid] = ((float)input_image[gid] / (float)PIXEL_COUNT) * 256;     // Y channel
	output_image[PIXEL_COUNT + gid] = input_image[PIXEL_COUNT + gid];             // Cb channel
	output_image[(2 * PIXEL_COUNT) + gid] = input_image[(2 * PIXEL_COUNT) + gid]; // Cr channel
	printf("Intensity1: %d, Intensity2: %d\n", input_image[gid], output_image[gid]);
}