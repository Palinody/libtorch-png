cmake_minimum_required(VERSION 3.0.2)
project(torch_png)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -march=native -Wall -Wfatal-errors)

find_package(
  catkin REQUIRED COMPONENTS
  roscpp
)
find_package(PCL REQUIRED)
find_package(Torch REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS 
        roscpp 
)

pkg_check_modules(LIBPNG libpng REQUIRED)

include_directories(
    include 
    ${catkin_INCLUDE_DIRS} 
    ${LIBPNG_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

add_library(
    ${PROJECT_NAME}
    src/Png.cpp
)

target_link_libraries(
    ${PROJECT_NAME} ${catkin_LIBRARIES}
)

# test mode: 
# $ catkin_make run_tests_<pkg name>
if(CATKIN_ENABLE_TESTING)
    include_directories(test)
    catkin_add_gmock(
        PngTests
        src/Png.cpp
        test/PngTest.cpp
    )
    target_link_libraries(
        PngTests
        ${TORCH_LIBRARIES}
        ${LIBPNG_LIBRARIES}
    )
endif()