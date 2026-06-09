#include "../src/cuda_api.hpp"

#include <cuda_runtime.h>
#include <math.h>

namespace pb_cuda_pointcloud
{
namespace detail
{
namespace
{

__global__ void filterLivoxKernel(
  const RawLivoxPoint * input, int point_count, RawPointXYZI * output, int * output_count,
  int n_scans, int point_filter_num, float blind_sq, float det_range_sq)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= 0 || i >= point_count) {
    return;
  }

  const RawLivoxPoint point = input[i];
  if (point.line >= n_scans) {
    return;
  }
  const uint8_t return_tag = point.tag & 0x30;
  if (!(return_tag == 0x10 || return_tag == 0x00)) {
    return;
  }
  if (point_filter_num > 1 && (i % point_filter_num) != 0) {
    return;
  }
  const float dist_sq = point.x * point.x + point.y * point.y + point.z * point.z;
  if (dist_sq < blind_sq || dist_sq > det_range_sq) {
    return;
  }
  const RawLivoxPoint previous = input[i - 1];
  if (
    fabsf(point.x - previous.x) <= 1e-7f && fabsf(point.y - previous.y) <= 1e-7f &&
    fabsf(point.z - previous.z) <= 1e-7f) {
    return;
  }

  const int out_index = atomicAdd(output_count, 1);
  output[out_index] = RawPointXYZI{
    point.x, point.y, point.z, static_cast<float>(point.reflectivity),
    static_cast<float>(point.offset_time) / 1000000.0f};
}

__global__ void transformKernel(
  const float * input_x, const float * input_y, const float * input_z, float * output_x,
  float * output_y, float * output_z, int point_count, const float * matrix)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const float x = input_x[i];
  const float y = input_y[i];
  const float z = input_z[i];
  output_x[i] = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
  output_y[i] = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
  output_z[i] = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
}

bool checkCuda(cudaError_t result)
{
  return result == cudaSuccess;
}

}  // namespace

bool cudaDeviceAvailableImpl(int device_id)
{
  int count = 0;
  if (!checkCuda(cudaGetDeviceCount(&count))) {
    return false;
  }
  return device_id >= 0 && device_id < count;
}

bool cudaFilterLivoxImpl(
  const RawLivoxPoint * input, int point_count, RawPointXYZI * output, int * output_count,
  int n_scans, int point_filter_num, float blind_sq, float det_range_sq, int device_id)
{
  if (!cudaDeviceAvailableImpl(device_id)) {
    return false;
  }
  cudaSetDevice(device_id);

  RawLivoxPoint * device_input = nullptr;
  RawPointXYZI * device_output = nullptr;
  int * device_count = nullptr;
  const size_t input_bytes = sizeof(RawLivoxPoint) * point_count;
  const size_t output_bytes = sizeof(RawPointXYZI) * point_count;
  if (
    !checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_input), input_bytes)) ||
    !checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_output), output_bytes)) ||
    !checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_count), sizeof(int)))) {
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_count);
    return false;
  }

  int zero = 0;
  bool ok =
    checkCuda(cudaMemcpy(device_input, input, input_bytes, cudaMemcpyHostToDevice)) &&
    checkCuda(cudaMemcpy(device_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
  if (ok) {
    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;
    filterLivoxKernel<<<blocks, threads>>>(
      device_input, point_count, device_output, device_count, n_scans, point_filter_num, blind_sq,
      det_range_sq);
    ok = checkCuda(cudaGetLastError()) && checkCuda(cudaDeviceSynchronize());
  }
  if (ok) {
    ok = checkCuda(cudaMemcpy(output_count, device_count, sizeof(int), cudaMemcpyDeviceToHost));
  }
  if (ok && *output_count > 0) {
    ok = checkCuda(cudaMemcpy(output, device_output, sizeof(RawPointXYZI) * (*output_count), cudaMemcpyDeviceToHost));
  }

  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(device_count);
  return ok;
}

bool cudaTransformPointCloudImpl(
  const float * input_x, const float * input_y, const float * input_z, float * output_x,
  float * output_y, float * output_z, int point_count, const float * matrix_row_major,
  int device_id)
{
  if (!cudaDeviceAvailableImpl(device_id)) {
    return false;
  }
  cudaSetDevice(device_id);

  float * dx = nullptr;
  float * dy = nullptr;
  float * dz = nullptr;
  float * dox = nullptr;
  float * doy = nullptr;
  float * doz = nullptr;
  float * dm = nullptr;
  const size_t bytes = sizeof(float) * point_count;
  bool ok =
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dx), bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dy), bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dz), bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dox), bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&doy), bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&doz), bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dm), sizeof(float) * 16));
  if (ok) {
    ok =
      checkCuda(cudaMemcpy(dx, input_x, bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemcpy(dy, input_y, bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemcpy(dz, input_z, bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemcpy(dm, matrix_row_major, sizeof(float) * 16, cudaMemcpyHostToDevice));
  }
  if (ok) {
    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;
    transformKernel<<<blocks, threads>>>(dx, dy, dz, dox, doy, doz, point_count, dm);
    ok = checkCuda(cudaGetLastError()) && checkCuda(cudaDeviceSynchronize());
  }
  if (ok) {
    ok =
      checkCuda(cudaMemcpy(output_x, dox, bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(output_y, doy, bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(output_z, doz, bytes, cudaMemcpyDeviceToHost));
  }

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dz);
  cudaFree(dox);
  cudaFree(doy);
  cudaFree(doz);
  cudaFree(dm);
  return ok;
}

}  // namespace detail
}  // namespace pb_cuda_pointcloud
