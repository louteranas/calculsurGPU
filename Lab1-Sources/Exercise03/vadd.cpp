/* ----------------------------------------------------------------
**
** Name:       vadd_cpp.cpp
**
** Purpose:    Elementwise addition of two vectors (c = a + b)
**
**                   c = a + b
**
** ----------------------------------------------------------------
*/

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library
#include "device_picker.hpp"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include <err_code.h>

// ----------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

int main(int argc, char *argv[])
{
  std::vector<float> h_a(LENGTH);                // a vector
  std::vector<float> h_b(LENGTH);                // b vector
  std::vector<float> h_c(LENGTH, 0xdeadbeef);    // c = a + b, from compute device

  cl::Buffer d_a;                        // device memory used for the input  a vector
  cl::Buffer d_b;                        // device memory used for the input  b vector
  cl::Buffer d_c;                       // device memory used for the output c vector

  // Fill vectors a and b with random float values
  int count = LENGTH;
  for(int i = 0; i < count; i++)
  {
    h_a[i]  = rand() / (float)RAND_MAX;
    h_b[i]  = rand() / (float)RAND_MAX;
  }

  try
  {
      cl_uint deviceIndex = 2;
      parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
            std::cout << "Invalid device index (try '--list')\n";
            return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);

    // Load in kernel source, creating a program object for the context
    cl::Program program(context, util::loadProgram("vadd.cl"), true);

    // Get the command queue
    cl::CommandQueue queue(context);

    // Create the kernel functor

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

    d_a  = cl::Buffer(context, h_a.begin(), h_a.end(), true);
    d_b  = cl::Buffer(context, h_b.begin(), h_b.end(), true);

    d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

    util::Timer timer;

    vadd( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_a, d_b, d_c, count);

    queue.finish();

    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    std::cout<<"The kernels ran in "<<rtime <<" seconds"<<std::endl;

    cl::copy(queue, d_c, h_c.begin(), h_c.end());

    // Test the results
    int correct = 0;
    float tmp;
    for(int i = 0; i < count; i++) {
      tmp = h_a[i] + h_b[i]; // expected value for d_c[i]
      tmp -= h_c[i];         // compute errors
      if(tmp*tmp < TOL*TOL) {
        // correct if square deviation is less
        correct++;  //  than tolerance squared
      }
      else {
        std::cout<<"tmp "<<tmp <<", h_a " << h_a[i]
          << ", h_b " << h_b[i] << ", h_c "<<h_c[i]<<std::endl;
      }
    }

    // summarize results
    std::cout<< "vector add to find C = A+B: " << correct <<" "
      << "out of "<<count<<"results were correct."<< std::endl;
  }
  catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what()
      << "(" << err_code(err.err()) << ")"
      << std::endl;
  }
}
