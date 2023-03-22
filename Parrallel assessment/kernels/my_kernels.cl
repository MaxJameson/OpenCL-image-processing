
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

kernel void histogram_Local(global const uchar* A, global uint* H, global uint* binsDivider, global uint* bins,local uint* LocalMem) {
	// gets the current index for global and local memeory
	int id = get_global_id(0);
	int lid = get_local_id(0);

	// Uses a barrier to sync local memeory loading
	barrier(CLK_LOCAL_MEM_FENCE);



	// gets the intensity value from the image
	const uchar pixel = A[id];
	float bin = (uint)pixel / (*binsDivider);
	uint location = round(bin);
	atomic_inc(&LocalMem[location]);
	barrier(CLK_LOCAL_MEM_FENCE);

	// uses an atomic function to increment the current intensity
	if(id < *bins){
		atomic_add(&H[id], LocalMem[id]);
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
