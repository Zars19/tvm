/*!
 * \file src/runtime/contrib/ft/ft.cc
 * \brief TVM compatible wrappers for FT-2500.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <pthread.h>

#include "ft.h"

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DTYPE)  \
      extern "C" void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out, int64_t p_DIM1_, int64_t p_DIM2_) {        \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                            \
          for (int64_t j = 0; j < p_DIM2_; ++j) {                          \
            int64_t k = i * p_DIM2_ + j;                                   \
            out[k] = a[k] p_OP_ b[k];                                      \
          }                                                                \
        }                                                                  \
      }

namespace tvm {
namespace runtime {
namespace contrib {

// should comply with src/relay/backend/contrib/ft/codegen.cc

CSOURCE_BINARY_OP_2D(ft_binary_op_add, +, float);

CSOURCE_BINARY_OP_2D(ft_binary_op_multiply, *, float);

CSOURCE_BINARY_OP_2D(ft_binary_op_subtract, -, float);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
