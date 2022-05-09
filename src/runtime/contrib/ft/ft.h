#ifndef TVM_RUNTIME_CONTRIB_FT_FT_H_
#define TVM_RUNTIME_CONTRIB_FT_FT_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace runtime {
namespace contrib {

extern "C" void ft_binary_op_add(float* a, float* b, float* out, int64_t p_SIZ_);

extern "C" void ft_binary_op_multiply(float* a, float* b, float* out, int64_t p_SIZ_);

extern "C" void ft_binary_op_subtract(float* a, float* b, float* out, int64_t p_SIZ_);

extern "C" void ft_unary_op_abs(float* a, float* out, int64_t p_SIZ_);

extern "C" void ft_unary_op_relu(float* a, float* out, int64_t p_SIZ_);

extern "C" void ft_unary_op_sigmod(float* a, float* out, int64_t p_SIZ_);

extern "C" void ft_unary_op_tanh(float* a, float* out, int64_t p_SIZ_);

extern "C" void ft_conv2d(float* data, float* weight, float* out, std::vector<int> nchw,
                          std::vector<int> oihw);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_FT_FT_H_
