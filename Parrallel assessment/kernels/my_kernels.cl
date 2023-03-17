
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

kernel void histogram_Local(global const uchar* A, global uint* H, global uint* binsDivider,local uchar* LocalMem, local uint* Localbin) {
	// gets the current index for global and local memeory
	int id = get_global_id(0);
	int lid = get_local_id(0);

	// stores the global data is local memory
	LocalMem[lid] = A[id];
	Localbin = *binsDivider;

	// Uses a barrier to sync local memeory loading
	barrier(CLK_GLOBAL_MEM_FENCE);



	// gets the intensity value from the image
	const uchar pixel = LocalMem[lid];
	float bin = (int)pixel / (int)Localbin;
	int location = round(bin);

	// uses an atomic function to increment the current intensity
	atomic_inc(&H[location]);
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

	A[id] = minScale + (A[id] - *min) * (maxScale - minScale) / (*max - *min);

}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void equalise( global uchar* in, global uchar* out,global uint* hist, global uint* binsDivider) {
	int id = get_global_id(0);
	int in_intensity = in[id] / *binsDivider;
	int new_intensity = 0;
	if (id == 0) { // perform this part only once i.e. for work item 0
		printf("currentSize: %d\n", get_local_size(0) - 1);
	}

	if(in_intensity == get_global_size(0) - 1){
		new_intensity = hist[in_intensity];
	}
	else{
		new_intensity = hist[in_intensity + 1];
	}

	out[id] = new_intensity;
}
