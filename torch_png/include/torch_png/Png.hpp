#pragma once

#include <png.h>
#include <torch/torch.h>

#include <filesystem>
#include <tuple>
/**
 * @brief libtorch handles only torch::UInt8 which are of depth 8.
 * Other bitdepths have not been considered when writting this code.
 *
 * ** LIBPNG ENCODING/DECODING **
 * Color    Allowed    Interpretation
 * Type    Bit Depths
 *
 *  0       1,2,4,8,16  Each pixel is a grayscale sample.
 *
 *  2       8,16        Each pixel is an R,G,B triple.
 *
 *  3       1,2,4,8     Each pixel is a palette index;
 *                     a PLTE chunk must appear.
 *
 *  4       8,16        Each pixel is a grayscale sample,
 *                     followed by an alpha sample.
 *
 *  6       8,16        Each pixel is an R,G,B triple,
 *                     followed by an alpha sample.
 *
 * ** VALID LIBTORCH ENCODING/DECODING **
 * Color    Allowed
 * Type     Bit Depths
 *  0           8       -> gray       [0, 255]
 *  2           8       -> rgb        [0, 255]
 *  4           8       -> gray alpha [0, 255]
 *  6           8       -> rgb alpha  [0, 255]
 */
namespace torch_png {

namespace fs  = std::filesystem;
namespace idx = torch::indexing;

// PNG_COLORS put in array so that channel index directly maps to corresponding color encoding
static constexpr std::uint8_t channel_idx_to_color[4] = {
    PNG_COLOR_TYPE_GRAY,       /*    0     */
    PNG_COLOR_TYPE_GRAY_ALPHA, /*    4     */
    PNG_COLOR_TYPE_RGB,        /*    2     */
    PNG_COLOR_TYPE_RGB_ALPHA   /*    6     */
};
/**
 * @brief Get the PNG infos:
 *      - height
 *      - width
 *      - channels
 *      - bit depth
 *      - color type
 *
 * @param filepath
 * @return std::tuple<std::int32_t, std::int32_t, std::uint8_t, std::uint8_t, std::uint8_t>
 */
std::tuple<std::int32_t, std::int32_t, std::uint8_t, std::uint8_t, std::uint8_t> getDims(const fs::path& filepath);
/**
 * @brief Reads a png file and returns a torch tensor
 * with dims {channels, height, width}
 *
 * @param filepath
 * @return 3D torch::Tensor
 */
torch::Tensor decode(const fs::path& filepath);
/**
 * @brief writes a png file from a torch tensor of dims {channels, height, width}
 *
 * @param filepath
 * @param torch_tensor 3D torch::Tensor
 */
void encode(const fs::path& filepath, const torch::Tensor& tensor);
/**
 * @brief Encodes a batch of images into a (sequence of) png files
 * if a batch dimension is provided, then the stem will be <stem> += "{delimiter}{#batch}" + <ext>
 *
 * @param filepath
 * @param tensor 4D torch::Tensor
 * @param delimiter can be any string, "_", "__", "-" except forbidden ones "/", ":", "." etc
 */
void encode_batch(fs::path filepath, const torch::Tensor& tensor, const std::string& delimiter = "_");

}  // namespace torch_png