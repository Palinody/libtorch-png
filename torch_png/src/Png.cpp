#include "torch_png/Png.hpp"

#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace torch_png {

namespace {

typedef std::unique_ptr<FILE, int (*)(FILE*)> unique_fp;
/**
 * @brief Smart file pointer handler
 *
 * @param filename
 * @param flags action to perform when closing the file
 * @return unique_fp
 */
unique_fp make_unique_fp(const char* filename, const char* flags) {
    return unique_fp(fopen(filename, flags), fclose);
}

/**
 * @brief @brief checks if the number of channels are valid
 *
 * @param channels number of channels in a torch::Tensor image
 * @return true
 * @return false
 */
bool is_valid_channels(std::int64_t channels) {
    if (channels == 4 || channels == 3 || channels == 2 || channels == 1)
        return true;
    return false;
}
/**
 * @brief Checks that tensor is valid and returns a PNG encoded copy
 *
 * @param tensor tensor to check and copy
 * @return torch::Tensor copy of tensor
 */
torch::Tensor check_transform_cpy(const torch::Tensor& tensor) {
    if (tensor.dim() != 3)
        throw std::invalid_argument("Unexpected torch::Tensor dimensions.\nGot(" + std::to_string(tensor.dim()) +
                                    "). Expects 3.");
    if (tensor.dtype() != torch::kUInt8)
        throw std::invalid_argument("Unexpected torch::Tensor type. Expects: torch::kUInt8");

    const auto channels = tensor.size(0);

    if (!is_valid_channels(channels))
        throw std::invalid_argument("Unexpected torch::Tensor channels.\nGot(" + std::to_string(channels) +
                                    "). Expects 1, 2, 3, 4.");
    // return to png format
    return tensor.detach().clone().permute({1, 2, 0}).to(torch::kCPU).contiguous();
}

}  // namespace

std::tuple<std::int32_t, std::int32_t, std::uint8_t, std::uint8_t, std::uint8_t> getDims(const fs::path& filepath) {
    char header[8];  // max size that can be checked
    // open and test if png file
    auto fp = make_unique_fp(filepath.c_str(), "rb");
    if (!fp.get())
        abort();

    if (!fread(header, 1, 8, fp.get()))
        abort();

    if (png_sig_cmp((png_const_bytep)header, 0, 8))
        abort();

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        abort();

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        abort();
    }
    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
        abort();
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        abort();
    }
    png_init_io(png_ptr, fp.get());
    // lets libpng know there are some bytes missing (the 8 we read)
    png_set_sig_bytes(png_ptr, 8);
    // read all the file information up to the actual image data
    png_read_info(png_ptr, info_ptr);

    const auto height     = png_get_image_height(png_ptr, info_ptr);
    const auto width      = png_get_image_width(png_ptr, info_ptr);
    const auto channels   = png_get_channels(png_ptr, info_ptr);
    const auto bit_depth  = png_get_bit_depth(png_ptr, info_ptr);
    const auto color_type = png_get_color_type(png_ptr, info_ptr);
    // const auto rowbytes = png_get_rowbytes(png_ptr, info_ptr);

    png_read_update_info(png_ptr, info_ptr);

    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

    return {height, width, channels, bit_depth, color_type};
}

torch::Tensor decode(const fs::path& filepath) {
    char header[8];  // max size that can be checked
    // open and test if png file
    auto fp = make_unique_fp(filepath.c_str(), "rb");
    if (!fp.get())
        abort();

    if (!fread(header, 1, 8, fp.get()))
        abort();

    if (png_sig_cmp((png_const_bytep)header, 0, 8))
        abort();

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        abort();

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        abort();
    }
    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
        abort();
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        abort();
    }
    png_init_io(png_ptr, fp.get());
    // lets libpng know there are some bytes missing (the 8 we read)
    png_set_sig_bytes(png_ptr, 8);

    // read all the file information up to the actual image data
    png_read_info(png_ptr, info_ptr);

    const auto height    = png_get_image_height(png_ptr, info_ptr);
    const auto width     = png_get_image_width(png_ptr, info_ptr);
    const auto channels  = png_get_channels(png_ptr, info_ptr);
    const auto bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    // Currently handles only bit_depth 8
    if (bit_depth != 8) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        abort();
    }
    png_read_update_info(png_ptr, info_ptr);
    // read file
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        abort();
    }

    auto options      = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto torch_tensor = torch::empty({height, width, channels}, options).contiguous();

    for (std::int64_t offset = 0; offset < height * width * channels; offset += width * channels)
        png_read_row(png_ptr, (torch_tensor.data_ptr<std::uint8_t>() + offset), NULL);

    png_read_end(png_ptr, end_info);

    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

    return torch_tensor.permute({2, 0, 1}).contiguous();
}

void encode(const fs::path& filepath, const torch::Tensor& tensor) {
    auto tensor_cpy = check_transform_cpy(tensor);

    auto fp = make_unique_fp(filepath.c_str(), "wb");
    if (!fp.get())
        abort();

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        abort();

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        abort();
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        abort();
    }
    png_init_io(png_ptr, fp.get());
    /* write header */
    if (setjmp(png_jmpbuf(png_ptr)))
        abort();

    const auto height   = tensor_cpy.size(0);
    const auto width    = tensor_cpy.size(1);
    const auto channels = tensor_cpy.size(2);

    png_set_IHDR(png_ptr,
                 info_ptr,
                 width,
                 height,
                 (CHAR_BIT * sizeof(std::uint8_t)),
                 channel_idx_to_color[channels - 1],
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    // write bytes
    if (setjmp(png_jmpbuf(png_ptr)))
        abort();

    for (std::int64_t offset = 0; offset < height * width * channels; offset += width * channels)
        png_write_row(png_ptr, (tensor_cpy.data_ptr<std::uint8_t>() + offset));

    // end write
    if (setjmp(png_jmpbuf(png_ptr)))
        abort();

    png_write_end(png_ptr, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);
}

void encode_batch(fs::path filepath, const torch::Tensor& tensor, const std::string& delimiter) {
    if (tensor.dim() != 4)
        throw std::invalid_argument("Unexpected torch::Tensor dim.\nGot(" + std::to_string(tensor.dim()) +
                                    "). Expects 4.");
    const auto batch = tensor.size(0);
    // save a copy of extension (may be empty string "")
    const auto ext = filepath.extension();
    // save a copy of (f)ile(p)ath without extension and remove extension from filepath
    const auto fp_no_ext = filepath.replace_extension("");
#ifdef _OPENMP
#pragma omp parallel for firstprivate(filepath)
#endif
    for (std::int64_t b = 0; b < batch; ++b) {
        // append index
        filepath += fs::path(delimiter + std::to_string(b));
        // and extension
        filepath += ext;
        // encode a single image at a time
        encode(filepath, tensor.index({b, idx::Ellipsis}));
        // reset path to raw path name without extension
        filepath = fp_no_ext;
    }
}

}  // namespace torch_png