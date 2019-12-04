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
#define LENGTH (16777216)    // length of vectors a, b, and c

int main(int argc, char *argv[])
{
  std::vector<float> h_a(LENGTH);                // a vector
  std::vector<float> h_b(LENGTH);                // b vector
  std::vector<float> h_c(LENGTH, 0xdeadbeef);    // c = a + b, from compute device
  std::vector<float> h_d(LENGTH);                // d = c + e vector
  std::vector<float> h_e(LENGTH);                // e vector
  std::vector<float> h_f(LENGTH);                // f = d + g, from compute device
  std::vector<float> h_g(LENGTH);

  cl::Buffer d_a;                        // device memory used for the input  a vector
  cl::Buffer d_b;                        // device memory used for the input  b vector
  cl::Buffer d_c;                       // device memory used for the output c vector
  cl::Buffer d_d;
  cl::Buffer d_e;
  cl::Buffer d_f;
  cl::Buffer d_g;


  // Fill vectors a and b with random float values
  int count = LENGTH;
  for(int i = 0; i < count; i++)
  {
    h_a[i]  = rand() / (float)RAND_MAX;
    h_b[i]  = rand() / (float)RAND_MAX;
    h_e[i]  = rand() / (float)RAND_MAX;
    h_g[i] = rand() / (float)RAND_MAX;
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
/*
    // Load in kernel source, creating a program object for the context
    cl::Program program(context, util::loadProgram("vadd.cl"), true);

    // Get the command queue
    cl::CommandQueue queue(context);

    // Create the kernel functor

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");


    util::Timer timer;

    //_____________

    d_a  = cl::Buffer(context, h_a.begin(), h_a.end(), true);
    d_b  = cl::Buffer(context, h_b.begin(), h_b.end(), true);

    d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

    vadd( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_a, d_b, d_c, count);

    queue.finish();

    cl::copy(queue, d_c, h_c.begin(), h_c.end());

    //_____________

    d_a  = cl::Buffer(context, h_c.begin(), h_c.end(), true);
    d_b  = cl::Buffer(context, h_e.begin(), h_e.end(), true);

    d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

    vadd( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_a, d_b, d_c, count);

    queue.finish();

    cl::copy(queue, d_c, h_d.begin(), h_d.end());
    //_____________

    d_a  = cl::Buffer(context, h_d.begin(), h_d.end(), true);
    d_b  = cl::Buffer(context, h_g.begin(), h_g.end(), true);

    d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);


    vadd( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_a, d_b, d_c, count);

    queue.finish();

    cl::copy(queue, d_c, h_f.begin(), h_f.end());

    //_____________


    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    std::cout<<"The kernels ran in "<<rtime <<" seconds"<<std::endl;




    // Test the results
    int correct = 0;
    float tmp;

    for(int i = 0; i < count; i++) {
      tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i]; // expected value for d_c[i]
      tmp -= h_f[i];         // compute errors
      if(tmp*tmp < TOL*TOL) {
          correct++;
      }
      else {
        std::cout<<"tmp "<<tmp <<", h_a " << h_a[i]
          << ", h_b " << h_b[i] << ", h_c "<<h_c[i]<<std::endl;
      }
    }

    // summarize results
    std::cout<< "vector add to find F = A + B + E + G: " << correct <<" "
      << "out of "<<count<<" results were correct."<< std::endl;
*/
//________________________________________________________________________

    std::cout << "Deuxième méthode :"<<std::endl;

    // Load in kernel source, creating a program object for the context
    cl::Program program(context, util::loadProgram("vaddBis.cl"), true);

    // Get the command queue
    cl::CommandQueue queue(context);

    // Create the kernel functor

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,int> vaddBis(program, "vaddBis");
    util::Timer timer2;

    //_____________

    d_a  = cl::Buffer(context, h_a.begin(), h_a.end(), true);
    d_b  = cl::Buffer(context, h_b.begin(), h_b.end(), true);
    d_c  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
    d_d  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
    d_e  = cl::Buffer(context, h_e.begin(), h_e.end(), true);
    d_f  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
    d_g  = cl::Buffer(context, h_g.begin(), h_g.end(), true);

    vaddBis( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_a, d_b, d_c, d_d, d_e, d_f, d_g, count);

    queue.finish();

    cl::copy(queue, d_c, h_c.begin(), h_c.end());
    cl::copy(queue, d_d, h_d.begin(), h_d.end());
    cl::copy(queue, d_f, h_f.begin(), h_f.end());

    double rtime = static_cast<double>(timer2.getTimeMilliseconds()) / 1000.0;
    std::cout<<"The kernels ran in "<<rtime <<" seconds"<<std::endl;
    int correct = 0;
    float tmp;

    for(int i = 0; i < count; i++) {
      tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i]; // expected value for d_c[i]
      tmp -= h_f[i];         // compute errors
      if(tmp*tmp < TOL*TOL) {
          correct++;
      }
      else {
        std::cout<<"tmp "<<tmp <<", h_a " << h_a[i]
          << ", h_b " << h_b[i] << ", h_c "<<h_c[i]<<std::endl;
      }
    }

  }
  catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what()
      << "(" << err_code(err.err()) << ")"
      << std::endl;
  }

}
