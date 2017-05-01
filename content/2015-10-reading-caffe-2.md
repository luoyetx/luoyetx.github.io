Title: Caffe 源码阅读 Blob
Date: 2015-10-31
Slug: reading-caffe-2
Category: Machine Learning


Blob 在 Caffe 中扮演了重要的角色，用于存储数据和网络参数，同时也在 CPU 和 GPU 之间做了数据同步。Blob 原本在 Caffe 中被表示为一个 4 维数组 (num x channel x height x width)，现在可以表示多维数组，最高维数由宏 `kMaxBlobAxes` 确定，目前 blob.hpp 中设置了 `const int kMaxBlobAxes = 32;`。Blob 类的代码主要集中在 blob.hpp 和 blob.cpp 中。

### 数据与相关操作函数

Blob 类主要包括如下成员

```cpp
shared_ptr<SyncedMemory> data_; // data 数据
shared_ptr<SyncedMemory> diff_; // diff 数据
shared_ptr<SyncedMemory> shape_data_; // 每一维数据的大小
vector<int> shape_; // 跟 shape_data_ 一样
int count_; // 当前容纳的数据大小
int capacity_; // 最大能够容纳的数据大小
```

其中 SyncedMemory 主要用来实现数据在 CPU 和 GPU 上的管理。同时 Blob 类提供一组函数来操作这些数据。

```cpp
const Dtype* cpu_data() const;
void set_cpu_data(Dtype* data);
const int* gpu_shape() const;
const Dtype* gpu_data() const;
const Dtype* cpu_diff() const;
const Dtype* gpu_diff() const;
Dtype* mutable_cpu_data();
Dtype* mutable_gpu_data();
Dtype* mutable_cpu_diff();
Dtype* mutable_gpu_diff();
```

我们可以通过这些函数拿到 Blob 内部的数据包括修改 Blob 的内部数据。其中的 Dtype 是泛型类型，在定义 Blob 变量时设置的，一般为 float 或者 double。

Blob 类在内部所存储的数据是一块连续的内存，为了表示多维数组，shape_ 和 shape_data_ 记录了每一维的大小，这样就能够很轻松地从给出的坐标中计算出 offset 从而得到那个点的数据。由于 Blob 主要还是用来表示 4 维数组 (最初就是这样的)，Blob 类中仍使用了 `int num(); int channels(); int height(); int width();` 这些函数，其实 num 等价于 shape()[0]，channels 等价于 shape()[1]，height 等价于 shape()[2]，width 等价于 shape()[3]。计算 offset 时可以使用这四个数字或者直接给出坐标。

```cpp
int offset(const int n, const int c = 0, const int h = 0, const int w = 0);
int offset(const vector<int>& indices);
```

有了 Blob 提供的这组函数和上一组函数，我们就可以轻易地操作 Blob 内部的数据了。

### 动态多维数组

Blob 类可以动态改变数组的尺寸，当拓展数组导致原有内存空间不足以存放下数据时 (count_ > capacity_)，就会重新分配内存。Blob 提供了一组 Reshape 函数来完成这个功能。

```cpp
void Reshape(const int num, const int channels, const int height, const int width); // Deprecated
void Reshape(const vector<int>& shape);
void Reshape(const BlobShape& shape);
void ReshapeLike(const Blob& other);
```

Blob 类在初始化时并没有分配内存，也是通过调用 Reshape 来分配内存的。

```cpp
template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes); // 检查维数
  count_ = 1; // 用于计算新的多维数组的大小
  shape_.resize(shape.size()); // 更新维数
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    // shape_data_ 未初始化或者内存太小
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    // 内存不够
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}
```

### SyncedMemory

Blob 事实上是对 SyncedMemory 的封装。SyncedMemory 完成了对内存的实际操作，包括数据在 CPU 和 GPU 上的同步。

```cpp
enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };

void* cpu_ptr_; // cpu 数据
void* gpu_ptr_; // gpu 数据
size_t size_; // 数据大小
SyncedHead head_; // 数据同步状态
bool own_cpu_data_; // 是否拥有当前 cpu 数据
bool cpu_malloc_use_cuda_; // 是否采用 CUDA 来分配 CPU 数据，默认不用
bool own_gpu_data_; // 是否拥有当前 gpu 数据
int gpu_device_; // gpu 数据所在的显卡号
```

SyncedMemory 内部存放了两份数据，分别位于 CPU 和 GPU 上，用 cpu_ptr 和 gpu_ptr 表示。同时 SyncedMemory 也给出了一组函数来获取和设置实际数据。

```cpp
const void* cpu_data();
void set_cpu_data(void* data);
const void* gpu_data();
void set_gpu_data(void* data);
void* mutable_cpu_data();
void* mutable_gpu_data();
```

head_ 表示了数据的同步状态，通过调用 `to_cpu()` 和 `to_gpu()` 来做同步。如果 head_ = UNINITIALIZED 则分配相应的内存。

```cpp
inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_); // 分配内存
    caffe_memset(size_, 0, cpu_ptr_); // 初始化为 0
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      // 如果未初始化，则分配内存
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    // 复制 GPU 数据到 CPU
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_)); // 获取显卡号
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_)); // 在指定显卡上分配内存
    caffe_gpu_memset(size_, 0, gpu_ptr_); // 初始化为 0
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      // 未初始化就在指定显卡上分配内存
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_); // 复制数据
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}
```

### 数据序列化

Blob 数据可以通过 Protobuf 来做相应的序列化操作，`ToProto` 和 `FromProto` 完成相应的序列化操作。

```protobuf
message BlobProto {
  optional BlobShape shape = 7;
  repeated float data = 5 [packed = true];
  repeated float diff = 6 [packed = true];
  repeated double double_data = 8 [packed = true];
  repeated double double_diff = 9 [packed = true];

  // 4D dimensions -- deprecated.  Use "shape" instead.
  optional int32 num = 1 [default = 0];
  optional int32 channels = 2 [default = 0];
  optional int32 height = 3 [default = 0];
  optional int32 width = 4 [default = 0];
}
```

### 小结

Caffe 通过 SyncedMemory 和 Blob 封装了底层数据，为 Caffe 框架上的其他组件提供最基础的数据抽象，后面的 Layer 参数，Net 参数以及 Solver 的参数等都是 Blob 数据，所以理解 Blob 抽象和管理数据的实现方式有助于后续 Caffe 源码的阅读，也是阅读 Caffe 源码的第一步。

### 参考资料

- [Caffe 源码](https://github.com/BVLC/caffe)
