
// Counts occurence of each intensity
kernel void histogram(global const uint* A, global uint* H, global uint* binsDivider) {
	
	// gets the current index
	int id = get_global_id(0);


	// gets the intensity value from the image and calculates it's bin
	uint pixel = A[id];
	float bin = (uint)pixel / (*binsDivider);
	uint location = round(bin);

	// prevents issues with 0 values diplicating to size of the image
	if(location != 0){
	
		// uses an atomic function to increment the current intensity
		atomic_inc(&H[location]);
	}

}

kernel void histogram_Local(global const uint* A, global uint* H, global uint* binsDivider,local uint* LocalMem) {
	// gets the current index for global and local memeory
	int id = get_global_id(0);

	// Uses a barrier to sync local memeory loading
	barrier(CLK_LOCAL_MEM_FENCE);

	if(id == 0){
		printf("local size %d\n", get_local_size(0));
	}

	// gets the intensity value from the image
	uint pixel = A[id];
	float bin = (uint)pixel / (*binsDivider);
	uint location = round(bin);

	// stores in local memeory
	atomic_inc(&LocalMem[location]);

	// syncs memeory
	barrier(CLK_LOCAL_MEM_FENCE);

	// stores in global memeory
	atomic_add(&H[location], LocalMem[location]);


}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void C_histogram(global  uint* A) {
	int id = get_global_id(0);
	int n = get_global_size(0);
	int t;

	// runs upsweep on vector
	for (int stride = 1; stride < n; stride *=2){
		if(((id+1) % (stride*2)) == 0){

			// passes values forward
			A[id] += A[id - stride];
		}
		// syncs memeory
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// runs downsweep on vector
	if (id == 0){
		A[n-1] = 0;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);


	for (int stride = n/2; stride > 0; stride /= 2){
		if(((id+1) % (stride*2)) == 0){

			// passes values forward
			t = A[id];
			A[id] += A[id - stride];
			A[id - stride] = t;
		}
		// syncs memeory
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//Hillis-Steele basic inclusive scan
kernel void C_histogramhs(global uint* A, global uint* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	// store data to prevent override
	global int* C;

	// loops through vector
	for (int stride = 1; stride <= N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride){
			// adds cumulative sum
			B[id] += A[id - stride];
		}

		// syncs memeory
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		// swaps values to prevent overwriting
		C = A; A = B; B = C; //swap A & B between steps
	}
}


//a simple OpenCL kernel which copies all pixels from A to B
kernel void N_histogram( global uint* A, global uint* min, global uint* max) {
	int id = get_global_id(0);

	// reduce size of value to prevent overflow
	int currentValue = A[id] / 10;
	// stores 0-1 normalisation
	double minScale = 0;
	double maxScale = 1;
	double normalised;
	if(id == 0 || A[id] == 0){
		// prevents need to calculate 0 count entries
		A[id] = 0;
	}
	else{
		// normlises entry between 0 - 1
		normalised = minScale + (currentValue - *min) * (maxScale - minScale) / (*max - *min);
		// scales normalistion to 0 - 255
		A[id] = normalised * 255;
	}


}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void equalise( global uint* in, global uint* out,global uint* hist, global uint* binsDivider) {
	int id = get_global_id(0);
	// calculates bin location
	int in_intensity = in[id] / *binsDivider;
	if(id == get_global_size(0)){
		// gets intensity
		out[id] = hist[in_intensity];
	}
	else{
		// gets intensity
		out[id] = hist[in_intensity + 1];
	}

}


kernel void reduce(global uint* A){
	int id = get_local_id(0);
	int N = get_local_size(0);

	// loops through vector
	for(int stride=1; stride<N; stride*=2){
		// stores minimum value
		if((id % (stride*2)) == 0){
			if(A[id] > A[id+stride] || A[id] == 0){
				A[id] = A[id+stride];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
