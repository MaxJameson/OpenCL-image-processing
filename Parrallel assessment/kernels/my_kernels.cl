
// Counts occurence of each intensity
kernel void histogram(global const uchar* A, global uint* H, global uint* binsDivider, global uint* bins) {
	
	// gets the current index
	int id = get_global_id(0);


	// gets the intensity value from the image
	const uchar pixel = A[id];
	float bin = (uint)pixel / (*binsDivider);
	int location = round(bin);

	// uses an atomic function to increment the current intensity
	atomic_inc(&H[location]);
}

kernel void histogram_Local(global const uchar* A, global uint* H, global uint* binsDivider, global uint* size,local uint* LocalMem) {
	// gets the current index for global and local memeory
	int id = get_global_id(0);

	// Uses a barrier to sync local memeory loading
	barrier(CLK_LOCAL_MEM_FENCE);

	if(id == 0){
		printf("local size %d\n", get_local_size(0));
	}

	// gets the intensity value from the image
	const uchar pixel = A[id];
	float bin = (uint)pixel / (*binsDivider);
	uint location = round(bin);
	//printf("work group size %d\n", &LocalMem[location]);
	atomic_inc(&LocalMem[location]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if((id%32) == 0){
		atomic_add(&H[location], LocalMem[location]);
	}

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

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void C_histogramhs(global uint* A, global uint* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride <= N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}


//a simple OpenCL kernel which copies all pixels from A to B
kernel void N_histogram( global uint* A, global uint* min, global uint* max) {
	int id = get_global_id(0);
	int currentValue = A[id] / 10;
	double minScale = 0;
	double maxScale = 1;
	double normalised;
	if(id == 0 || A[id] == 0){
		A[id] = 0;
	}
	else{
		normalised = minScale + (currentValue - *min) * (maxScale - minScale) / (*max - *min);
		A[id] = normalised * 255;
	}


}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void equalise( global uchar* in, global uchar* out,global uint* hist, global uint* binsDivider) {
	int id = get_global_id(0);
	int in_intensity = in[id] / *binsDivider;
	int new_intensity = 0;

	if(id == 0){
		new_intensity = hist[in_intensity + 1];
	}
	else{
		new_intensity = hist[in_intensity -1];
	}

	out[id] = new_intensity;
}
