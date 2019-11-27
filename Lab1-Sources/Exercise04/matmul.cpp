/* ----------------------------------------------------------------
**
**  PROGRAM: Matrix Multiplication driver
**
**  PURPOSE: This is a driver program to test various ways of computing
**           the product:
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**  USAGE:   The matrices are constant matrices, square and the order is
**           set as a constant, ORDER (see mult.h).
**
** ----------------------------------------------------------------
*/

#include "matmul.hpp"
#include "matrix_lib.hpp"
#include "util.hpp"
#include <err_code.h>
#include "device_picker.hpp"

int main(int argc, char *argv[])
{

    int N;    // A[N][N], B[N][N], C[N][N]
    int size; // Number of elements in each matrix

    double start_time; // Starting time
    double run_time;   // Timing
    util::Timer timer; // Timing

    N = ORDER;
    size = N * N;

    std::vector<float> h_A(size); // Host memory for Matrix A
    std::vector<float> h_B(size); // Host memory for Matrix B
    std::vector<float> h_C(size); // Host memory for Matrix C

    cl::Buffer d_a, d_b, d_c; // Matrices in device memory

    // ------------------------------------------------------------------
    // Create a context and queue
    // ------------------------------------------------------------------

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
        cl::CommandQueue queue(context, device);

        // ------------------------------------------------------------------
        // Run sequential matmul
        // ------------------------------------------------------------------

        initmat(N, h_A, h_B, h_C);

        timer.reset();

        std::cout << "\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======" << N << std::endl;
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            seq_mat_mul_sdot(N, h_A, h_B, h_C);

            run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
            results(N, h_C, run_time);
        }

        // ------------------------------------------------------------------
        // Setup the buffers, initialize matrices, and write them into global memory
        // ------------------------------------------------------------------

        //  Reset A, B and C matrices (just to play it safe)
        initmat(N, h_A, h_B, h_C);

        d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);

        d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);

        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

        // ------------------------------------------------------------------
        // OpenCL matrix multiplication ... Naive
        // ------------------------------------------------------------------

        timer.reset();
        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("matmul.cl"), true);

        // Create the compute kernel from the program
        cl::Kernel kernel_mul = cl::Kernel(program, "mmul");

        // Display max group size for execution
        std::cout << "\nWork Group Size " << kernel_mul.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << std::endl;
        std::cout << "Work Group Memory size " << kernel_mul.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device) << std::endl;

        std::cout << "\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======" << N << std::endl;

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            // Set output matrix to 0
            zero_mat(N, h_C);

            // Initialize arguments of kernel
            kernel_mul.setArg(0, N);
            kernel_mul.setArg(1, d_a);
            kernel_mul.setArg(2, d_b);
            kernel_mul.setArg(3, d_c);

            // Set workspace and workgroup topologies
            cl::NDRange global(N, N);
            cl::NDRange local(16, 16);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            // Execute the kernel over the entire range of C matrix elements ... computing
            // a dot product for each element of the product matrix.  The local work
            // group size is set to NULL ... so I'm telling the OpenCL runtime to
            // figure out a local work group size for me.

            queue.enqueueNDRangeKernel(kernel_mul, cl::NullRange, global, local);

            queue.finish();

            run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;

            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            results(N, h_C, run_time);

        } // end for loop
    }
    catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
