# libtorch-png

A ros package tested on ros-melodic for encoding/decoding png images with libtorch `torch::Tensor`.

**This package is intended to be compiled with ros-melodic and catkin but you may as well just take the `include/torch_png/Png.hpp` and `src/Png.cpp` files and compile and link however you want.**

## Requirements

- c++17
- libtorch
- libpng

## Optional

- ros-melodic

## Add it to you ros workspace (example)

```ssh
$ cd catkin_ws/src/external_libraries
$ git clone https://github.com/Palinody/libtorch-png
```

## Compiling

If you are using ros:
```ssh
$ catkin_make
```

## Using the ros package library from another ros package (minimal example)

```ssh
find_package(
  catkin REQUIRED COMPONENTS
  # <...>
  roscpp
  torch_png
)

target_link_libraries(
    ${PROJECT_NAME} 
    # <...>
    ${LIBPNG_LIBRARIES}
    ${TORCH_LIBRARIES}
)
```

## Unit tests

```ssh
$ catkin_make run_tests_torch_png
```

The `test/PngTest.cpp` file is ugly because its content aggregates the content of files from a greater project. It has been included in this repository just to show how the package is intended to be used and how it has been tested. The images are being stored in `/tmp` and immediately deleted upon creation.

## Supported image formats
`torch::kUInt` type only.
- gray
- gray alpha
- rgb
- rgb alpha

## How to use

```c
#include "torch_png/Png.hpp"

int main(int argc, char** argv) {
  // returns a torch::UInt8 with dims {channels, height, width}
  const auto image = torch_png::decode("path/to/dir/file.png");
  // expects a torch::UInt8 with dims {channels, height, width}
  torch_png::encode("path/to/dir/file.png", image);
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
  // expects a torch::UInt8 with dims {batch, channels, height, width}
  torch_png::encode_batch("path/to/dir/filename.png", batched_tensor);
  return 0;
}
```