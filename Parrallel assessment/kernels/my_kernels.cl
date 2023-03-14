
//a simple OpenCL kernel which copies all pixels from A to B
kernel void histogram(global const uchar* A, global uint* H) {
	int id = get_global_id(0);
	const uchar pixel = A[id];
	const int bin = (int)pixel;

	atomic_inc(&H[bin]);
}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void C_histogram(global  uint* A) {
	int id = get_global_id(0);
	int n = get_global_size(0);
	int t;

	for (int stride = 1; stride < n; stride *=2){

		if(((id+1) % (stride*2)) == 0){
			A[id] += A[id - stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if(id == 0){
		A[n-1] = 0;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int stride = n/2; stride > 0; stride /=2){

		if(((id+1) % (stride*2)) == 0){
			t = A[id];
			A[id] += A[id - stride];
			A[id - stride] = t;
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void N_histogram( global uint* A, global uint* min, global uint* max) {
	int id = get_global_id(0);
	uint minScale = 0;
	uint maxScale = 255;

	if (id == 0){
		printf("Max %d\n", *max);
		printf("Min %d\n", *min);
	}

	A[id] = minScale + (A[id] - *min) * (maxScale - minScale) / (*max - *min);

}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void equalise( global uchar* in, global uchar* out,global uint* hist) {
	int id = get_global_id(0);
	int in_intensity = in[id];
	int new_intensity = 0;

	if(id == get_global_size(0)){
		new_intensity = hist[in_intensity];
	}
	else{
		new_intensity = hist[in_intensity + 1];
	}

	out[id] = new_intensity;
}



kernel void invert(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);

	unsigned char intensity =  A[id];

    // Adjust the intensity value
    intensity = (unsigned char) (255 - A[id]);

	if (id == 0){
		printf("Size: %d\n", get_global_size(0));
	}

	int stop = (get_global_size(0) / 3) * 2;

	if(id <= 30){
		B[id] = 0;
	}
	else{
	    // Write the new intensity value back to the buffer
		B[id] = intensity;
	}


}