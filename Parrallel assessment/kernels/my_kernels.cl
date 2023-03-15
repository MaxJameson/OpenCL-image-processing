
// Counts occurence of each intensity
kernel void histogram(global const uchar* A, global uint* H) {
	
	// gets the current index
	int id = get_global_id(0);

	// gets the intensity value from the image
	const uchar pixel = A[id];
	const int bin = (int)pixel;

	// uses an atomic function to increment the current intensity
	atomic_inc(&H[bin]);
}

kernel void histogram_Local(global const uchar* A, global uint* H, local uchar* LocalMem) {
	// gets the current index for global and local memeory
	int id = get_global_id(0);
	int lid = get_local_id(0);

	// stores the global data is local memory
	LocalMem[lid] = A[id];

	// Uses a barrier to sync local memeory loading
	barrier(CLK_GLOBAL_MEM_FENCE);

	// gets the intensity value from the image
	const uchar pixel = LocalMem[lid];
	const int bin = (int)pixel;

	// Uses a barrier to sync intensity calculation
	barrier(CLK_GLOBAL_MEM_FENCE);

	// uses an atomic function to increment the current intensity
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
	uint maxScale = 256;

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
