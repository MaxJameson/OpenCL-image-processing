
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

// locally counts the occurence of each intensity
kernel void histogram_Local( global const int * A, global int * H,global uint* binsDivider,local int * LH) {
	// gets index values
	int id = get_global_id(0); int lid = get_local_id(0);



	// gets the intensity value from the image and calculates it's bin
	uint pixel = A[id];
	float bin = (uint)pixel / (*binsDivider);
	uint location = round(bin);

	// stores in a local histogram
	atomic_inc(&LH[location]);
	
	// sync local memeory
	barrier(CLK_LOCAL_MEM_FENCE);

	// combines local histograms
	if (id < 256){
		atomic_add(&H[id], LH[id]);
	}

}


// cumulative histogram using Blelloch scan in global memory
kernel void C_histogram(global  uint* A) {
	
	// gets index values
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

	// sync memory
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

// cumulative histogram using Blelloch scan in local memory
kernel void C_histogram_Local(global  uint* A, local uint* l) {
	
	// get index values
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int n = get_local_size(0);
	int t;

	// passes global memory to local
	l[lid] = A[id];

	// syncs memeory
	barrier(CLK_LOCAL_MEM_FENCE);

	// runs upsweep on vector
	for (int stride = 1; stride < n; stride *=2){
		if(((lid+1) % (stride*2)) == 0){

			// passes values forward
			l[lid] += l[lid - stride];
		}
		// syncs memeory
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// runs downsweep on vector
	if (lid == 0){
		l[n-1] = 0;
	}

	// syncs memeory
	barrier(CLK_LOCAL_MEM_FENCE);


	for (int stride = n/2; stride > 0; stride /= 2){
		if(((lid+1) % (stride*2)) == 0){

			// passes values forward
			t = l[lid];
			l[lid] += l[lid - stride];
			l[lid - stride] = t;
		}
		// syncs memeory
		barrier(CLK_LOCAL_MEM_FENCE);
	}



	// adds to local memory to global
	atomic_xchg(&A[id], l[lid]);
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

//Hillis-Steele basic inclusive scan
kernel void C_histogramhs_Local(global uint* A, global uint* B, local uint* lA, local uint* lB) {
	int id = get_global_id(0);
	int N = get_local_size(0);
	int lid = get_local_id(0);

	lA[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// store data to prevent override
	local int* C;

	// loops through vector
	for (int stride = 1; stride <= N; stride *= 2) {
		lB[lid] = lA[lid];
		if (lid >= stride){
			// adds cumulative sum
			lB[lid] += lA[lid - stride];
		}

		// syncs memeory
		barrier(CLK_LOCAL_MEM_FENCE); //sync the step

		// swaps values to prevent overwriting
		C = lA; lA = lB; lB = C; //swap A & B between steps
	}

	// adds local memory to global
	atomic_xchg(&B[id], lB[lid]);
}


//a simple OpenCL kernel which copies all pixels from A to B
kernel void N_histogram( global uint* A, global uint* min, global uint* max, global uint* bits) {
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
		A[id] = normalised * (*bits - 1);
	}


}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void equalise( global uint* in, global uint* out,global uint* hist, global uint* binsDivider) {
	int id = get_global_id(0);
	// calculates bin location
	int in_intensity = in[id] / *binsDivider;

	// prevents final value from reaching outide scope
	if(id == get_global_size(0)){
		// gets intensity
		out[id] = hist[in_intensity];
	}
	else{
		// accounts for Blelloch scan being inclusive
		out[id] = hist[in_intensity + 1];
	}

}


kernel void reduce(global uint* A){
	int id = get_local_id(0);
	int N = get_local_size(0);

	// loops through vector
	for(int stride=1; stride<N; stride*=2){
		if((id % (stride*2)) == 0){
			
			// checks if stride number is lower than new number
			if(A[id] > A[id+stride] && A[id+stride] != 0){

				// updates new value with new minimum
				A[id] = A[id+stride];
			}
		}
		// syncs memeory
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
