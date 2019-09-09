#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernel_code.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef float mytype;

		//Part 3 - memory allocation
		//host - input
		//Importing the last column from the dataset
		std::cout << "...READING DATA (PLEASE WAIT)... " << std::endl;
		std::vector<mytype> temperatures = {};
		ifstream readFile("temp_lincolnshire.txt");
		string line;
		while (getline(readFile, line)) {
			vector<string> lines_vect = {};
			istringstream ext_line(line);
			while (getline(ext_line, line, ' ')) {
				lines_vect.push_back(line);
			}
			temperatures.push_back(std::stof(lines_vect.back()));
		}
		size_t non_padded_input_elements = temperatures.size();

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 1024;

		size_t padding_size = temperatures.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<mytype> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			temperatures.insert(temperatures.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = temperatures.size();//number of input elements
		size_t input_size = temperatures.size() * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		int num_outputs = 1;
		std::vector<mytype> mean(num_outputs);
		std::vector<mytype> min(num_outputs);
		std::vector<mytype> max(num_outputs);
		size_t output_size = num_outputs * sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_mean(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_min(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_max(context, CL_MEM_READ_WRITE, output_size);

		//ALL THE EVENTS FOR PERFORMANCE MEASURMENT FOR ALL THE OPERATIONS BEFORE STANDARD DEVIATION
		//UPLOAD EVENTS
		cl::Event _input_u;
		cl::Event _mean_u;
		cl::Event _min_u;
		cl::Event _max_u;
		//OPERATION EVENTS
		cl::Event _mean_o;
		cl::Event _min_o;
		cl::Event _max_o;
		//DOWNLOAD EVENTS
		cl::Event _mean_d;
		cl::Event _min_d;
		cl::Event _max_d;

		//Part 4 - device operations
		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, input_size, &temperatures[0], NULL, &_input_u);
		queue.enqueueFillBuffer(buffer_mean, 0, 0, output_size, NULL, &_mean_u);
		queue.enqueueFillBuffer(buffer_min, 0, 0, output_size, NULL, &_min_u);
		queue.enqueueFillBuffer(buffer_max, 0, 0, output_size, NULL, &_max_u);

		//4.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_mean = cl::Kernel(program, "reduce_sum");
		kernel_mean.setArg(0, buffer_input);
		kernel_mean.setArg(1, buffer_mean);
		kernel_mean.setArg(2, cl::Local(local_size * sizeof(mytype)));
		//kernel_mean.setArg(3, offset);
		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
		kernel_min.setArg(0, buffer_input);
		kernel_min.setArg(1, buffer_min);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));
		cl::Kernel kernel_max = cl::Kernel(program, "reduce_max");
		kernel_max.setArg(0, buffer_input);
		kernel_max.setArg(1, buffer_max);
		kernel_max.setArg(2, cl::Local(local_size * sizeof(mytype)));

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &_mean_o);
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &_min_o);
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &_max_o);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_mean, CL_TRUE, 0, output_size, &mean[0], NULL, &_mean_d);
		//The mean kernel simply produces a single value of sum with reduction. The output is extracted and divided by N to get the mean constant required for standard deviation.
		mean[0] = mean[0] / non_padded_input_elements;
		queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0, output_size, &min[0], NULL, &_min_d);
		queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0, output_size, &max[0], NULL, &_max_d);

		//ALL THE EVENTS FOR PERFORMANCE MEASURMENT FOR ALL THE OPERATIONS FOR VARIANCE
		//UPLOAD EVENTS
		cl::Event _variance_u;
		cl::Event _mean_2_u;
		//OPERATION EVENTS
		cl::Event _variance_o;
		//DOWNLOAD EVENTS
		cl::Event _variance_d;

		//VARIANCE
		std::vector<mytype> variance(input_elements);
		cl::Buffer buffer_variance(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_mean_2(context, CL_MEM_READ_ONLY, output_size);
		queue.enqueueFillBuffer(buffer_variance, 0, 0, input_size, NULL, &_variance_u);
		queue.enqueueWriteBuffer(buffer_mean_2, CL_TRUE, 0, output_size, &mean[0], NULL, &_mean_2_u);

		cl::Kernel kernel_variance = cl::Kernel(program, "map_variance");
		kernel_variance.setArg(0, buffer_input);
		kernel_variance.setArg(1, buffer_mean_2);
		kernel_variance.setArg(2, buffer_variance);

		queue.enqueueNDRangeKernel(kernel_variance, cl::NullRange, cl::NDRange(input_elements), cl::NullRange, NULL, &_variance_o);
		queue.enqueueReadBuffer(buffer_variance, CL_TRUE, 0, input_size, &variance[0], NULL, &_variance_d);
		variance.resize(non_padded_input_elements); // getting rid of the instances where x - mean(x) where x is 0 because of boundaries is performed.
		variance.resize(input_elements, 0.0f); // resizing again to enable cache usage

		//ALL THE EVENTS FOR PERFORMANCE MEASURMENT FOR ALL THE OPERATIONS FOR STANDARD DEVIATION
		//UPLOAD EVENTS
		cl::Event _std_u;
		cl::Event _variance_2_u;
		//OPERATION EVENTS
		cl::Event _kernel_sum_o;
		//DOWNLOAD EVENTS
		cl::Event _std_d;

		//STANDARD DEVIATION
		std::vector<mytype> std(num_outputs);
		cl::Buffer buffer_std(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_variance_2(context, CL_MEM_READ_WRITE, input_size);
		queue.enqueueFillBuffer(buffer_std, 0, 0, output_size, NULL, &_std_u);
		queue.enqueueWriteBuffer(buffer_variance_2, CL_TRUE, 0, input_size, &variance[0], NULL, &_variance_2_u);


		cl::Kernel kernel_sum = cl::Kernel(program, "reduce_sum");
		kernel_sum.setArg(0, buffer_variance_2);
		kernel_sum.setArg(1, buffer_std);
		kernel_sum.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &_kernel_sum_o);
		queue.enqueueReadBuffer(buffer_std, CL_TRUE, 0, output_size, &std[0], NULL, &_std_d);

		//the extracted value from the summed variance is simply divided by N and square rooted to perform last two operations of standard deviation
		std[0] = sqrt(std[0]/non_padded_input_elements);

		std::cout << "STATISTICAL INFORMATION" << std::endl;
		std::cout << "Average: " << mean[0] << std::endl;
		std::cout << "Min: " << min[0] << std::endl;
		std::cout << "Max: " << max[0] << std::endl;
		std::cout << "STD: " << std[0] << std::endl;

		// PERFORMANCE CALCULATIONS (DEPRESSING CODE):
		//ALL THE UPLOAD EVENT CALCS:
		auto input_u = _input_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _input_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto mean_u = _mean_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _mean_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto min_u = _min_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _min_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto max_u = _max_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _max_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto variance_u = _variance_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _variance_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto mean_2_u = _mean_2_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _mean_2_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto std_u = _std_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _std_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto variance_2_u = _variance_2_u.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _variance_2_u.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto total_upload = input_u + mean_u + min_u + max_u + variance_u + mean_2_u + variance_2_u;
		// ALL THE OPERATION EVENT CALCS
		auto mean_o = _mean_o.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _mean_o.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto min_o = _min_o.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _min_o.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto max_o = _max_o.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _max_o.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto variance_o = _variance_o.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _variance_o.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto kernel_sum_o = _kernel_sum_o.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _kernel_sum_o.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto total_operation = mean_o + min_o + variance_o + kernel_sum_o;
		// ALL THE DOWNLOAD EVENT CALCS
		auto mean_d = _mean_d.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _mean_d.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto min_d = _min_d.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _min_d.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto max_d = _max_d.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _max_d.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto variance_d = _variance_d.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _variance_d.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto std_d = _std_d.getProfilingInfo<CL_PROFILING_COMMAND_END>() - _std_d.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto total_download = mean_d + min_d + max_d + variance_d + std_d;
		// SUMMING UP ALL PERFORmANCES
		auto complete_performance = total_upload + total_operation + total_download;

		//PRINTING OUT ALL THE PERFORMANCE METRICS CALCULATED ABOVE
		std::cout << "PERFORMANCE INFORMATION (ALL OPERATIONS COMBINED)" << std::endl;
		std::cout << "Work goup size: " << local_size << std::endl;
		std::cout << "Work items (padded): " << input_elements << std::endl;
		std::cout << "Total upload time (NS): " << total_upload << std::endl;
		std::cout << "Total operation time (NS): " << total_operation << std::endl;
		std::cout << "Total download time (NS): " << total_download << std::endl;
		std::cout << "TOTAL TIME (START-END): " << complete_performance << std::endl;


	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	return 0;
}
