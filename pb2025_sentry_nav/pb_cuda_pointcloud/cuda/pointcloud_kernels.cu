#include "../src/cuda_api.hpp"

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <vector>

namespace pb_cuda_pointcloud
{
namespace detail
{
namespace
{
constexpr unsigned long long kEmptyVoxelKey = 0xffffffffffffffffULL;

__global__ void filterLivoxKernel(
  const RawLivoxPoint * input, int point_count, RawPointXYZI * output, int * output_count,
  int n_scans, int point_filter_num, float blind_sq, float det_range_sq)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
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
  if (point_filter_num > 1 && ((i + 1) % point_filter_num) != 0) {
    return;
  }
  const float dist_sq = point.x * point.x + point.y * point.y + point.z * point.z;
  if (dist_sq < blind_sq || dist_sq > det_range_sq) {
    return;
  }
  if (
    fabsf(point.x - point.previous_x) <= 1e-7f &&
    fabsf(point.y - point.previous_y) <= 1e-7f &&
    fabsf(point.z - point.previous_z) <= 1e-7f) {
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

__device__ bool packVoxelKey(int x, int y, int z, unsigned long long * key)
{
  constexpr int bias = 1 << 20;
  constexpr int min_coord = -bias;
  constexpr int max_coord = bias - 2;
  if (x < min_coord || x > max_coord || y < min_coord || y > max_coord || z < min_coord ||
      z > max_coord) {
    return false;
  }
  constexpr unsigned long long mask = (1ULL << 21) - 1ULL;
  const unsigned long long ux = static_cast<unsigned long long>(x + bias) & mask;
  const unsigned long long uy = static_cast<unsigned long long>(y + bias) & mask;
  const unsigned long long uz = static_cast<unsigned long long>(z + bias) & mask;
  *key = (ux << 42) | (uy << 21) | uz;
  return true;
}

__device__ unsigned int hashVoxelKey(unsigned long long key)
{
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdULL;
  key ^= key >> 33;
  key *= 0xc4ceb9fe1a85ec53ULL;
  key ^= key >> 33;
  return static_cast<unsigned int>(key);
}

__global__ void voxelAccumKernel(
  const RawPointXYZI * input, int point_count, unsigned long long * keys, float * sum_x,
  float * sum_y, float * sum_z, float * sum_intensity, float * sum_curvature, int * counts,
  int table_size, float inverse_leaf)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }

  const RawPointXYZI point = input[i];
  if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z)) {
    return;
  }

  const int vx = static_cast<int>(floorf(point.x * inverse_leaf));
  const int vy = static_cast<int>(floorf(point.y * inverse_leaf));
  const int vz = static_cast<int>(floorf(point.z * inverse_leaf));
  unsigned long long key = 0;
  if (!packVoxelKey(vx, vy, vz, &key)) {
    return;
  }
  const int mask = table_size - 1;
  int slot = static_cast<int>(hashVoxelKey(key)) & mask;

  for (int probe = 0; probe < table_size; ++probe) {
    const unsigned long long old = atomicCAS(&keys[slot], kEmptyVoxelKey, key);
    if (old == kEmptyVoxelKey || old == key) {
      atomicAdd(&sum_x[slot], point.x);
      atomicAdd(&sum_y[slot], point.y);
      atomicAdd(&sum_z[slot], point.z);
      atomicAdd(&sum_intensity[slot], point.intensity);
      atomicAdd(&sum_curvature[slot], point.curvature);
      atomicAdd(&counts[slot], 1);
      return;
    }
    slot = (slot + 1) & mask;
  }
}

__global__ void laserScanKernel(
  const float * input_x, const float * input_y, const float * input_z,
  const float * input_intensity, int point_count, float * ranges, int ranges_size,
  float angle_min, float angle_max, float angle_increment, float min_height, float max_height,
  float min_intensity, float max_intensity, float range_min, float range_max)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }

  const float x = input_x[i];
  const float y = input_y[i];
  const float z = input_z[i];
  const float intensity = input_intensity[i];
  if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
    return;
  }
  if (z > max_height || z < min_height) {
    return;
  }
  if (intensity < min_intensity || intensity > max_intensity) {
    return;
  }
  const float range = hypotf(x, y);
  if (range < range_min || range > range_max) {
    return;
  }
  const float angle = atan2f(y, x);
  if (angle < angle_min || angle > angle_max) {
    return;
  }
  const int index = static_cast<int>((angle - angle_min) / angle_increment);
  if (index < 0 || index >= ranges_size) {
    return;
  }

  int * address = reinterpret_cast<int *>(&ranges[index]);
  int old = *address;
  int assumed;
  do {
    assumed = old;
    const float old_value = __int_as_float(assumed);
    if (old_value <= range) {
      break;
    }
    old = atomicCAS(address, assumed, __float_as_int(range));
  } while (assumed != old);
}

bool checkCuda(cudaError_t result)
{
  return result == cudaSuccess;
}

int cachedCudaDeviceCount()
{
  static const int count = []() {
    int device_count = 0;
    return checkCuda(cudaGetDeviceCount(&device_count)) ? device_count : 0;
  }();
  return count;
}

int nextPowerOfTwo(int value)
{
  if (value <= 1) {
    return 1;
  }
  int result = 1;
  while (result < value) {
    if (result > (1 << 29)) {
      return 0;
    }
    result <<= 1;
  }
  return result;
}

}  // namespace

bool cudaDeviceAvailableImpl(int device_id)
{
  const int count = cachedCudaDeviceCount();
  return device_id >= 0 && device_id < count;
}

bool cudaFilterLivoxImpl(
  const RawLivoxPoint * input, int point_count, RawPointXYZI * output, int * output_count,
  int n_scans, int point_filter_num, float blind_sq, float det_range_sq, int device_id)
{
  if (output_count != nullptr) {
    *output_count = 0;
  }
  if (
    input == nullptr || output == nullptr || output_count == nullptr || point_count <= 0 ||
    n_scans <= 0 || point_filter_num <= 0 || blind_sq < 0.0f || det_range_sq <= blind_sq ||
    !cudaDeviceAvailableImpl(device_id)) {
    return false;
  }
  if (!checkCuda(cudaSetDevice(device_id))) {
    return false;
  }

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
    ok = checkCuda(cudaMemcpy(
      output, device_output, sizeof(RawPointXYZI) * (*output_count), cudaMemcpyDeviceToHost));
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
  if (
    input_x == nullptr || input_y == nullptr || input_z == nullptr || output_x == nullptr ||
    output_y == nullptr || output_z == nullptr || matrix_row_major == nullptr ||
    point_count <= 0 || !cudaDeviceAvailableImpl(device_id)) {
    return false;
  }
  if (!checkCuda(cudaSetDevice(device_id))) {
    return false;
  }

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

bool cudaVoxelDownsampleImpl(
  const RawPointXYZI * input, int point_count, RawPointXYZI * output, int * output_count,
  float leaf_size, int device_id)
{
  if (output_count != nullptr) {
    *output_count = 0;
  }
  if (
    input == nullptr || output == nullptr || output_count == nullptr || point_count <= 0 ||
    point_count > (1 << 28) || leaf_size <= 0.0f || !cudaDeviceAvailableImpl(device_id)) {
    return false;
  }
  if (!checkCuda(cudaSetDevice(device_id))) {
    return false;
  }

  const int table_size = nextPowerOfTwo(point_count * 2);
  if (table_size <= 0) {
    return false;
  }
  RawPointXYZI * device_input = nullptr;
  unsigned long long * device_keys = nullptr;
  float * device_sum_x = nullptr;
  float * device_sum_y = nullptr;
  float * device_sum_z = nullptr;
  float * device_sum_intensity = nullptr;
  float * device_sum_curvature = nullptr;
  int * device_counts = nullptr;

  const size_t input_bytes = sizeof(RawPointXYZI) * point_count;
  const size_t table_float_bytes = sizeof(float) * table_size;
  const size_t table_int_bytes = sizeof(int) * table_size;
  const size_t table_key_bytes = sizeof(unsigned long long) * table_size;

  std::vector<unsigned long long> keys;
  std::vector<float> sum_x;
  std::vector<float> sum_y;
  std::vector<float> sum_z;
  std::vector<float> sum_intensity;
  std::vector<float> sum_curvature;
  std::vector<int> counts;
  try {
    keys.resize(table_size);
    sum_x.resize(table_size);
    sum_y.resize(table_size);
    sum_z.resize(table_size);
    sum_intensity.resize(table_size);
    sum_curvature.resize(table_size);
    counts.resize(table_size);
  } catch (...) {
    return false;
  }

  bool ok =
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_input), input_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_keys), table_key_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_sum_x), table_float_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_sum_y), table_float_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_sum_z), table_float_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_sum_intensity), table_float_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_sum_curvature), table_float_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_counts), table_int_bytes));

  if (ok) {
    ok =
      checkCuda(cudaMemcpy(device_input, input, input_bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemset(device_keys, 0xff, table_key_bytes)) &&
      checkCuda(cudaMemset(device_sum_x, 0, table_float_bytes)) &&
      checkCuda(cudaMemset(device_sum_y, 0, table_float_bytes)) &&
      checkCuda(cudaMemset(device_sum_z, 0, table_float_bytes)) &&
      checkCuda(cudaMemset(device_sum_intensity, 0, table_float_bytes)) &&
      checkCuda(cudaMemset(device_sum_curvature, 0, table_float_bytes)) &&
      checkCuda(cudaMemset(device_counts, 0, table_int_bytes));
  }

  if (ok) {
    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;
    voxelAccumKernel<<<blocks, threads>>>(
      device_input, point_count, device_keys, device_sum_x, device_sum_y, device_sum_z,
      device_sum_intensity, device_sum_curvature, device_counts, table_size, 1.0f / leaf_size);
    ok = checkCuda(cudaGetLastError()) && checkCuda(cudaDeviceSynchronize());
  }

  if (ok) {
    ok =
      checkCuda(cudaMemcpy(keys.data(), device_keys, table_key_bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(sum_x.data(), device_sum_x, table_float_bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(sum_y.data(), device_sum_y, table_float_bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(sum_z.data(), device_sum_z, table_float_bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(
        sum_intensity.data(), device_sum_intensity, table_float_bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(
        sum_curvature.data(), device_sum_curvature, table_float_bytes, cudaMemcpyDeviceToHost)) &&
      checkCuda(cudaMemcpy(counts.data(), device_counts, table_int_bytes, cudaMemcpyDeviceToHost));
    if (ok) {
      int count = 0;
      for (int i = 0; i < table_size; ++i) {
        if (keys[i] == kEmptyVoxelKey || counts[i] <= 0) {
          continue;
        }
        const float inv_count = 1.0f / static_cast<float>(counts[i]);
        output[count++] = RawPointXYZI{
          sum_x[i] * inv_count, sum_y[i] * inv_count, sum_z[i] * inv_count,
          sum_intensity[i] * inv_count, sum_curvature[i] * inv_count};
      }
      *output_count = count;
    }
  }

  cudaFree(device_input);
  cudaFree(device_keys);
  cudaFree(device_sum_x);
  cudaFree(device_sum_y);
  cudaFree(device_sum_z);
  cudaFree(device_sum_intensity);
  cudaFree(device_sum_curvature);
  cudaFree(device_counts);
  return ok;
}

bool cudaPointCloudToLaserScanImpl(
  const float * input_x, const float * input_y, const float * input_z,
  const float * input_intensity, int point_count, float * ranges, int ranges_size,
  float angle_min, float angle_max, float angle_increment, float min_height, float max_height,
  float min_intensity, float max_intensity, float range_min, float range_max, int device_id)
{
  if (
    input_x == nullptr || input_y == nullptr || input_z == nullptr ||
    input_intensity == nullptr || ranges == nullptr || point_count <= 0 || ranges_size <= 0 ||
    angle_increment <= 0.0f || angle_max <= angle_min || range_min < 0.0f ||
    range_max <= range_min || !cudaDeviceAvailableImpl(device_id)) {
    return false;
  }
  if (!checkCuda(cudaSetDevice(device_id))) {
    return false;
  }

  float * dx = nullptr;
  float * dy = nullptr;
  float * dz = nullptr;
  float * di = nullptr;
  float * dranges = nullptr;
  const size_t point_bytes = sizeof(float) * point_count;
  const size_t ranges_bytes = sizeof(float) * ranges_size;
  bool ok =
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dx), point_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dy), point_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dz), point_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&di), point_bytes)) &&
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&dranges), ranges_bytes));

  if (ok) {
    ok =
      checkCuda(cudaMemcpy(dx, input_x, point_bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemcpy(dy, input_y, point_bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemcpy(dz, input_z, point_bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemcpy(di, input_intensity, point_bytes, cudaMemcpyHostToDevice)) &&
      checkCuda(cudaMemcpy(dranges, ranges, ranges_bytes, cudaMemcpyHostToDevice));
  }
  if (ok) {
    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;
    laserScanKernel<<<blocks, threads>>>(
      dx, dy, dz, di, point_count, dranges, ranges_size, angle_min, angle_max, angle_increment,
      min_height, max_height, min_intensity, max_intensity, range_min, range_max);
    ok = checkCuda(cudaGetLastError()) && checkCuda(cudaDeviceSynchronize());
  }
  if (ok) {
    ok = checkCuda(cudaMemcpy(ranges, dranges, ranges_bytes, cudaMemcpyDeviceToHost));
  }

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dz);
  cudaFree(di);
  cudaFree(dranges);
  return ok;
}

}  // namespace detail
}  // namespace pb_cuda_pointcloud
