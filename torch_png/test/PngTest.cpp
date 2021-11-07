#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "torch_png/Png.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <initializer_list>
#include <vector>

#include <sstream>

namespace torch_typing {

/**
 * @brief
 *
 * @tparam T
 */
template <typename T>
void printType(const T&) {
    std::cout << __PRETTY_FUNCTION__;
}
/**
 * @brief returns the torch version of std type
 *
 * @tparam T
 * @return constexpr torch::Dtype
 */
template <typename T>
constexpr torch::Dtype std_to_torch_type() {
    if constexpr (std::is_same_v<T, float>)
        return torch::kFloat32;
    else if constexpr (std::is_same_v<T, double>)
        return torch::kFloat64;
    else if constexpr (std::is_same_v<T, std::int8_t>)
        return torch::kInt8;
    else if constexpr (std::is_same_v<T, std::int16_t>)
        return torch::kInt16;
    else if constexpr (std::is_same_v<T, std::int32_t>)
        return torch::kInt32;
    else if constexpr (std::is_same_v<T, std::int64_t>)
        return torch::kInt64;
    else if constexpr (std::is_same_v<T, std::uint8_t>)
        return torch::kUInt8;
    else
        throw std::invalid_argument("Bad type");
}
/**
 * @brief Get the Type object
 *
 * @param bitdepth
 * @return torch::Dtype
 */
inline torch::Dtype getType(std::uint8_t bitdepth, float) {
    if (bitdepth == 32)
        return torch::kFloat32;
    else if (bitdepth == 64)
        return torch::kFloat64;
    else
        throw std::invalid_argument("Bitdepth for float must be 32 or 64.");
}
/**
 * @brief Get the Type object
 *
 * @param bitdepth
 * @return torch::Dtype
 */
inline torch::Dtype getType(std::uint8_t bitdepth, int) {
    if (bitdepth == 8)
        return torch::kInt8;
    else if (bitdepth == 16)
        return torch::kInt16;
    else if (bitdepth == 32)
        return torch::kInt32;
    else if (bitdepth == 64)
        return torch::kInt64;
    else
        throw std::invalid_argument("Bitdepth for int must be 8, 16, 32 or 64.");
}
/**
 * @brief Get the Type object
 *
 * @param bitdepth
 * @return torch::Dtype
 */
inline torch::Dtype getType(std::uint8_t bitdepth, unsigned) {
    if (bitdepth == 8)
        return torch::kUInt8;
    else
        throw std::invalid_argument("Bitdepth for unsigned int must be 8.");
}
/**
 * @brief Get the Type object
 *
 * @tparam T
 * @param bitdepth
 * @return torch::Dtype
 */
template <typename T>
inline torch::Dtype getType(std::uint8_t bitdepth) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, unsigned>) {
        static constexpr T var{};
        return getType(bitdepth, var);
    }
    throw std::invalid_argument("Unexpected type " + std::string(typeid(T).name()) + ".");
}

}  // namespace torch_typing

namespace torch_create {

/**
 * @brief makes a new tensor based on input values and dimensions
 *
 * @tparam T restricted type to match types that torch library handles
 * @param values vector of any size, nelements must match "total tensor volume"
 * @param dims result tensor length w.r.t. each dimension
 * @param device torch::kCPU or torch::kCUDA
 * @return torch::Tensor values tensor reshaped w.r.t. dims
 */
template <typename T,
          std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::int8_t> ||
                               std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::int32_t> ||
                               std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint8_t>,
                           bool> = true>
torch::Tensor make_tensor_values(std::vector<T>                             values,
                                 const std::initializer_list<std::int64_t>& dims,
                                 const torch::Device&                       device = torch::kCPU) {
    // ...
    const auto nelem = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::int64_t>());
    if (nelem != static_cast<std::int64_t>(values.size()))
        throw std::invalid_argument("The number of elements specified by dims in " + std::string(__func__) +
                                    " should match the number of elements of the data vector.\n" +
                                    std::to_string(nelem) + "!=" + std::to_string(values.size()));
    values.resize(nelem);
    return torch::from_blob(values.data(),
                            dims,
                            torch::TensorOptions()
                                .dtype(torch_typing::std_to_torch_type<T>())
                                .layout(torch::kStrided)
                                .device(torch::kCPU))
        .clone()
        .to(device);
}

}  // namespace torch_create

namespace fs = std::filesystem;

class PngErrorsTest : public ::testing::Test {
  protected:
    // where the png tests will be stored (and deleted)
    const fs::path fp = "/tmp";
};

TEST_F(PngErrorsTest, testTestingPath) {
    EXPECT_TRUE(fs::exists(fp));
}

TEST_F(PngErrorsTest, testReadWritePngTensorsRGB) {
    // create a {channels=3, rows=1, columns=3} rgb image
    const auto image_rgb =
        torch_create::make_tensor_values<std::uint8_t>({255, 0, 100, 100, 0, 255, 255, 255, 255}, {3, 1, 3});
    // encode it
    torch_png::encode(fp / "rgb.png", image_rgb);
    // decode it
    const auto image_rgb2 = torch_png::decode(fp / "rgb.png");
    // compare them
    EXPECT_TRUE(image_rgb2.eq(image_rgb).all().item<bool>());
    // delete the test image
    fs::remove(fp / "rgb.png");
}

TEST_F(PngErrorsTest, testReadWritePngTensorsRGBA) {
    // create a {channels=4, rows=1, columns=3} rgba image
    const auto image_rgba =
        torch_create::make_tensor_values<std::uint8_t>({255, 0, 100, 100, 0, 255, 0, 100, 255, 0, 100, 255}, {4, 1, 3});
    // encode it
    torch_png::encode(fp / "rgba.png", image_rgba);
    // decode it
    const auto image_rgba2 = torch_png::decode(fp / "rgba.png");
    // compare them
    EXPECT_TRUE(image_rgba2.eq(image_rgba).all().item<bool>());
    // delete the test image
    fs::remove(fp / "rgba.png");
}

TEST_F(PngErrorsTest, testReadWritePngTensorsGray) {
    // image gray
    const auto image_g = torch_create::make_tensor_values<std::uint8_t>({100, 0, 255}, {1, 1, 3});
    // encode it
    torch_png::encode(fp / "g.png", image_g);
    // decode it
    const auto image_g2 = torch_png::decode(fp / "g.png");
    // compare them
    EXPECT_TRUE(image_g2.eq(image_g).all().item<bool>());
    // delete the test image
    fs::remove(fp / "g.png");
}

TEST_F(PngErrorsTest, testReadWritePngTensorsGrayAlpha) {
    // image gray alpha
    const auto image_ga = torch_create::make_tensor_values<std::uint8_t>({100, 100, 100, 100, 0, 255}, {2, 1, 3});
    // encode it
    torch_png::encode(fp / "ga.png", image_ga);
    // decode it
    const auto image_ga2 = torch_png::decode(fp / "ga.png");
    // compare them
    EXPECT_TRUE(image_ga2.eq(image_ga).all().item<bool>());
    // delete the test image
    fs::remove(fp / "ga.png");
}

TEST_F(PngErrorsTest, testReadWritePngBatchedTensorsGray) {
    // batched image gray
    const auto image_g = torch_create::make_tensor_values<std::uint8_t>({100, 0, 255, 255, 0, 100}, {2, 1, 1, 3});
    // encode batch
    torch_png::encode_batch(fp / "g.png", image_g, "_");
    // decode first batch image
    const auto image_g2_first = torch_png::decode(fp / "g_0.png");
    // decode second batch image
    const auto image_g2_second = torch_png::decode(fp / "g_1.png");
    // make a batched tensor from decoded images
    const auto batched_tensor = torch::empty({2, 1, 1, 3});
    // copy contend from decoded images in batched tensor
    batched_tensor.index({0, torch_png::idx::Ellipsis}) = image_g2_first;
    batched_tensor.index({1, torch_png::idx::Ellipsis}) = image_g2_second;
    // compare decoded batchedtensor to the one we created
    EXPECT_TRUE(batched_tensor.eq(image_g).all().item<bool>());
    // delete the test images
    fs::remove(fp / "g_0.png");
    fs::remove(fp / "g_1.png");
}

TEST_F(PngErrorsTest, testExceptions) {
    // bad/good type
    const auto bad_tensor_type  = torch_create::make_tensor_values<float>({3, 2, 1}, {1, 1, 3});
    const auto good_tensor_type = torch_create::make_tensor_values<std::uint8_t>({3, 2, 1}, {1, 1, 3});
    EXPECT_THROW(torch_png::encode(fp / "bt0.png", bad_tensor_type), std::invalid_argument);
    EXPECT_NO_THROW(torch_png::encode(fp / "bt0.png", good_tensor_type));
    fs::remove(fp / "bt0.png");
    // wrong/right number of channels
    const auto tensor_c5 = torch_create::make_tensor_values<std::uint8_t>({1, 2, 3, 4, 5}, {5, 1, 1});
    const auto tensor_c4 = torch_create::make_tensor_values<std::uint8_t>({1, 2, 3, 4}, {4, 1, 1});
    const auto tensor_c3 = torch_create::make_tensor_values<std::uint8_t>({1, 2, 3}, {3, 1, 1});
    const auto tensor_c2 = torch_create::make_tensor_values<std::uint8_t>({1, 2}, {2, 1, 1});
    const auto tensor_c1 = torch_create::make_tensor_values<std::uint8_t>({1}, {1, 1, 1});
    EXPECT_THROW(torch_png::encode(fp / "c5.png", tensor_c5), std::invalid_argument);
    EXPECT_NO_THROW(torch_png::encode(fp / "c4.png", tensor_c4));
    fs::remove(fp / "c4.png");
    EXPECT_NO_THROW(torch_png::encode(fp / "c3.png", tensor_c3));
    fs::remove(fp / "c3.png");
    EXPECT_NO_THROW(torch_png::encode(fp / "c2.png", tensor_c2));
    fs::remove(fp / "c2.png");
    EXPECT_NO_THROW(torch_png::encode(fp / "c1.png", tensor_c1));
    fs::remove(fp / "c1.png");
    // calling (not) batched tensor with wrong function
    const auto batched_tensor =
        torch_create::make_tensor_values<std::uint8_t>({100, 0, 255, 255, 0, 100}, {2, 1, 1, 3});
    const auto not_batched_tensor = torch_create::make_tensor_values<std::uint8_t>({100, 0, 255}, {1, 1, 3});
    EXPECT_THROW(torch_png::encode(fp / "batched.png", batched_tensor), std::invalid_argument);
    EXPECT_THROW(torch_png::encode_batch(fp / "not_batched.png", not_batched_tensor), std::invalid_argument);
    EXPECT_NO_THROW(torch_png::encode(fp / "not_batched.png", not_batched_tensor));
    fs::remove(fp / "not_batched.png");
    EXPECT_NO_THROW(torch_png::encode_batch(fp / "batched.png", batched_tensor));
    fs::remove(fp / "batched_0.png");
    fs::remove(fp / "batched_1.png");
}

// catkin_make run_tests_ml
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}