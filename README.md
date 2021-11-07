# libtorch-png

A ros package tested on ros-melodic for encoding/decoding png images with libtorch `torch::Tensor`.

## Requirements

- c++17
- libtorch
- libpng

## Optional

- ros-melodic

This package is intended to be compiled with ros-melodic and catkin but you may as well just take the `include/torch_png/Png.hpp` and `src/Png.cpp` files and compile and link however you want.

## Compiling

```ssh
$ catkin_make
```

## Unit tests

```ssh
$ catkin_make run_tests_torch_png
```

The `test/PngTest.cpp` file is ugly because its content aggregates the content of files from a greater project. It has been included in this repository just to show how the package is intended to be used and how it has been tested. 

## How to use

```c
#include "torch_png/Png.hpp"

int main(int argc, char** argv) {
  const auto tensor = torch_png::decode("path/to/dir/file.png");
  // ...
  torch_png::encode("path/to/dir/file.png");
  // ...
  const auto batched_tensor = /*some batched tensor with TYPE torch::UInt8 and DIMS {batch, channels, height, width}*/
  /*
   * If your tensor is a batch of images, 
   * a sequence of png images will be written
   * Let batch = 3 and pathname = "path/to/dir/filename.png"
   * torch_png::encode_batch(...) will write batch files sequentially:
   *    "path/to/dir/filename_0.png"
   *    "path/to/dir/filename_1.png"
   *    "path/to/dir/filename_2.png"
   *
   * You may as well choose your own delimiter instead of "_", which is the default one.
   */
  torch_png::encode_batch("path/to/dir/filename.png", batched_tensor);
  return 0;
}
```