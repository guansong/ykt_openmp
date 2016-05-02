
// For clarity, error checking has been added.

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

#define BFS
//#define BFB

/* convert the kernel file into a string */
int convertToString(const char *filename, char*& str, int& size)
{
  //size_t size;
  //char*  str;
  std::fstream f(filename, (std::fstream::in | std::fstream::binary));

  if(f.is_open())
  {
    size_t fileSize;
    f.seekg(0, std::fstream::end);
    size = fileSize = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);
    str = new char[size+1];
    if(!str)
    {
      f.close();
      return 0;
    }

    f.read(str, fileSize);
    f.close();
    str[size] = '\0';
    return 0;
  }
  cout<<"Error: failed to open file\n:"<<filename<<endl;
  return -1;
}

template<typename T>
const char*
getOpenCLErrorCodeStr(T input)
{
    int errorCode = (int)input;
    switch(errorCode)
    {
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
             return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case CL_PLATFORM_NOT_FOUND_KHR:
            return "CL_PLATFORM_NOT_FOUND_KHR";
        //case CL_INVALID_PROPERTY_EXT:
        //    return "CL_INVALID_PROPERTY_EXT";
        case CL_DEVICE_PARTITION_FAILED_EXT:
            return "CL_DEVICE_PARTITION_FAILED_EXT";
        case CL_INVALID_PARTITION_COUNT_EXT:
            return "CL_INVALID_PARTITION_COUNT_EXT";
        default:
            return "unknown error code";
    }

    return "unknown error code";
}

#define CHK(t) \
{ \
  cl_int tmp = (t);\
  if (tmp != CL_SUCCESS) { \
    cout << "[Error:" << getOpenCLErrorCodeStr(t) << "] "\
      << __FILE__ << ":" << __LINE__ << ": " << #t \
       << endl; \
  }\
}

int main(int argc, char* argv[])
{

  enum format{
    src,
    bin
  };

  format fileformat = src;

  int i=0;
  printf("\ncmdline args count=%d", argc);

  /* First argument is executable name only */
  printf("\nexe name=%s", argv[0]);

  for (i=1; i< argc; i++) {
    printf("\narg%d=%s", i, argv[i]);
    if (!strncmp(argv[i], "-s", sizeof("-s"))) {
      printf(" src");
      fileformat = src;
    }
    if (!strncmp(argv[i], "-b", sizeof("-b"))) {
      printf(" bin");
      fileformat = bin;
    }
  }

  printf("\n");

  /*Step1: Getting platforms and choose an available one.*/
  cl_uint numPlatforms;//the NO. of platforms
  cl_platform_id platform = NULL;//the chosen platform
  cl_int  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS) {
    cout<<"Error: Getting platforms!"<<endl;
    return 1;
  }
  else {
    cout <<"Number of platforms: " << numPlatforms << endl;
  }

  /*For clarity, choose the first available platform. */
  if(numPlatforms > 0)
  {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    CHK(status = clGetPlatformIDs(numPlatforms, platforms, NULL));
    platform = platforms[0];
    delete[] platforms;
  }

  /*Step 2:Query the platform and choose the first GPU device if has one. Otherwise use the CPU as device.*/
  cl_uint       numDevices = 0;
  cl_device_id        *devices;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  if (numDevices == 0) //no GPU available.
  {
    cout << "No GPU device available."<<endl;
    cout << "Choose CPU as default device."<<endl;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
    devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

    CHK(status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL));
  }
  else
  {
    cout <<"Number of GPU device: " << numDevices << endl;
    devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
    CHK(status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL));
  }

  for (int i = 0; i < numDevices; i++) {
    char* value;
    size_t valueSize;
    // print device name
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, valueSize, value, NULL);
    cout <<"Device " << i << ": " << value << endl;
    //printf("%d. Device: %sn", j+1, value);
    if (value != NULL)
      free(value);
  }

  int deviceNum = 0;
  cout <<"  Using device: " << deviceNum << endl;
  cout << endl;

  /*Step 3: Create context.*/
  cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

  /*Step 4: Creating command queue associate with the context.*/
  cl_command_queue commandQueue = clCreateCommandQueue(context, devices[deviceNum], 0, NULL);

  /*Step 5: Create program object */
  const char *filename = "driver.cl";

  if ( fileformat == src )
    filename = "driver.cl";

  if ( fileformat == bin )
    filename = "driver.bin";

  char* source;

  int size;
  status = convertToString(filename, source, size);
  size_t sourceSize[] = {size};
  cl_int errcode;


  cl_program program;

  if (fileformat == src )
    program = clCreateProgramWithSource(context, 1, (const char**)&source, sourceSize, NULL);
  if (fileformat == bin )
    program = clCreateProgramWithBinary(context, 1, devices, sourceSize, (const unsigned char**)&source, &status , &errcode);

  CHK(status);
  
  /*Step 6: Build program. */
  cout << "Building ..." << endl << flush;
  CHK(status=clBuildProgram(program, 1, devices, "-cl-std=CL2.0 ", NULL, NULL));
  //CHK(status=clBuildProgram(program, 1, devices, "-save-temps-all -g -O0", NULL, NULL));
  cout << "finished." << endl << flush;

  {
    size_t len = 0;
    CHK(status=clGetProgramBuildInfo(program, devices[deviceNum], CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
    if (len > 1) {
      char *buffer = (char *)calloc(len, sizeof(char));
      CHK(status=clGetProgramBuildInfo(program, devices[deviceNum], CL_PROGRAM_BUILD_LOG, len, buffer, NULL));
      cout << "Build log: (" << len << ")" << endl << buffer << endl << flush;
    }
    if (status != CL_SUCCESS) {
      exit(1);
    }
  }

  /*Step 7: Initial input,output for the host and create memory objects for the kernel*/
  const char* input = "GdkknVnqkc";
  size_t strlength = strlen(input);

  cout<<"input string:"<<endl;
  cout<<input<<endl;

  char *output = (char*) calloc(strlength + 1, sizeof(char));
  cout<<"output string:"<<endl;
  cout<<output<<endl;

  cl_mem inputBuffer = clCreateBuffer(context,
      CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
      (strlength + 1) * sizeof(char),
      (void *) input,
      NULL);

  cl_mem outputBuffer = clCreateBuffer(context,
      CL_MEM_WRITE_ONLY,
      (strlength + 1) * sizeof(char),
      NULL,
      NULL);

  /*Step 8: Create kernel object */
  cl_kernel kernel = clCreateKernel(program,"helloworld", NULL);

  /*Step 9: Sets Kernel arguments.*/
  CHK(status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer));
  CHK(status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer));

  /*Step 10: Running the kernel.*/
  size_t global_work_size[1] = {strlength};
  CHK(status = 
      clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, global_work_size, 0, NULL, NULL));

  /*Step 11: Read the cout put back to host memory.*/
  CHK(status = 
      clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, strlength * sizeof(char), output, 0, NULL, NULL));

  output[strlength] = '\0';//Add the terminal character to the end of output.
  cout<<"output string:"<<endl;
  cout<<output<<endl;

  /*Step 12: Clean the resources.*/
  CHK(status = clReleaseKernel(kernel));//*Release kernel.
  CHK(status = clReleaseProgram(program)); //Release the program object.
  CHK(status = clReleaseMemObject(inputBuffer));//Release mem object.
  CHK(status = clReleaseMemObject(outputBuffer));
  CHK(status = clReleaseCommandQueue(commandQueue));//Release  Command queue.
  CHK(status = clReleaseContext(context));//Release context.

  if (output != NULL)
  {
    free(output);
    output = NULL;
  }

  if (devices != NULL)
  {
    free(devices);
    devices = NULL;
  }
  return 0;
}

