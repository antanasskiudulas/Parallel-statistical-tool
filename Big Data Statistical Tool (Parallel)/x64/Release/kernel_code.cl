//This implementation of float add was inspired by ANCA HAMURARU's article: Atomic operations for floats in OpenCL – improved
kernel void float_atomic_add(volatile global float *address, float value) {
	//unions allow for variables to hold multiple types
	//this allows for the global argument address pointer to be casted to integer pointer required for atomic_cmpxchg operation
    union { unsigned int int_value; float float_value;} readback;
    union { unsigned int int_value; float float_value;} old;
    do {
        old.float_value = *address;
        readback.float_value = old.float_value + value;
    } while (atomic_cmpxchg((volatile global unsigned int *)address, old.int_value, readback.int_value) != old.int_value);
}

kernel void float_atomic_max(volatile global float *address, float value) {
	union { unsigned int int_value; float float_value; } readback;
	union { unsigned int int_value; float float_value; } old;
	do {
		readback.float_value = *address;
		old.float_value = max(readback.float_value, value);
	} while (atomic_cmpxchg((volatile global unsigned int *)address, readback.int_value, old.int_value) != old.int_value);
}

kernel void float_atomic_min(volatile global float *address, float value) {
	union { unsigned int int_value; float float_value; } readback;
	union { unsigned int int_value; float float_value; } old;
	do {
		readback.float_value = *address;
		old.float_value = min(readback.float_value, value);
	} while (atomic_cmpxchg((volatile global unsigned int *)address, readback.int_value, old.int_value) != old.int_value);
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
kernel void reduce_sum(global const float* input, global float* output, local float* buffer) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	buffer[lid] = input[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			buffer[lid] += buffer[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		//atomicAdd_2(&output[0], buffer[lid]);
		float_atomic_add(&output[0], buffer[lid]);
	}
}


kernel void reduce_mean(global const float* input, global float* output, local float* buffer) {
	int id = get_global_id(0);
	int g_N = get_global_size(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	buffer[lid] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			buffer[lid] += buffer[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid) {
		int iterations = N*(g_N/N);
		float_atomic_add(&output[0], buffer[lid]/iterations);
	}
}

kernel void reduce_min(global const float* input, global float* output, local float* buffer) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	buffer[lid] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N) && buffer[lid] > buffer[lid + i]) 
			buffer[lid] = buffer[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid) {
		float_atomic_min(&output[0], buffer[lid]);
	}
}

kernel void reduce_max(global const float* input, global float* output, local float* buffer) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	buffer[lid] = input[id];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N) && buffer[lid] < buffer[lid + i]) 
			buffer[lid] = buffer[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid) {
			float_atomic_max(&output[0], buffer[lid]);
	}
}

kernel void map_variance(global const float* input, global const float* mean, global float* output) {
	int id = get_global_id(0);
	int g_size = get_global_size(0);
	output[id] = pow(input[id]-*mean,2);
}



