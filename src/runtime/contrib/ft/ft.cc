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
#include <math.h>

#include "ft.h"

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_)  \
      extern "C" void p_ID_(float* a, float* b, float* out, int64_t p_SIZ_) {        \
        for (int64_t i = 0; i < p_SIZ_; ++i) {                            \
            out[i] = a[i] p_OP_ b[i];                                     \
        }                                                                 \
      }

    #define CSOURCE_BINARY_OP(p_ID_, p_OP_)  \
      void *p_ID_##_thread(void *arg0) {               \
        binary_op_arg my_arg = *(binary_op_arg*)arg0; \
        for (unsigned i = 0; i < my_arg.len; ++i)     \
          my_arg.c[i] = my_arg.a[i] p_OP_ my_arg.b[i];\
        return NULL;                                  \
      }                                               \
      extern "C" void ft_binary_op_##p_ID_(float* a, float* b, float* out, int64_t p_SIZ_) {  \
        unsigned thread, block_size, start;                                               \
        pthread_t *thread_handles;                                                        \
        binary_op_arg *args;                                                                        \
        thread_handles =(pthread_t *)malloc(thread_count * sizeof(pthread_t));            \
        args = (binary_op_arg*)malloc(thread_count * sizeof(binary_op_arg));                                  \
        for (thread = 0, start = 0; thread < thread_count; ++thread) {                    \
          block_size = p_SIZ_ / thread_count + (thread < p_SIZ_ % thread_count);          \
          args[thread] = binary_op_arg(a + start, b + start, out + start, block_size);              \
              pthread_create(&thread_handles[thread], NULL,                               \
              p_ID_##_thread, (void *)&args[thread]);                                      \
          start += block_size ;                                                           \
        }                                                                                 \
        for (thread = 0; thread < thread_count; ++thread)                                 \
          pthread_join(thread_handles[thread], NULL);                                     \
        free(thread_handles);                                                             \
        free(args);                                                                       \
      }


namespace tvm {
namespace runtime {
namespace contrib {

// should comply with src/relay/backend/contrib/ft/codegen.cc

// CSOURCE_BINARY_OP_2D(ft_binary_op_add, +, float);

// CSOURCE_BINARY_OP_2D(ft_binary_op_multiply, *, float);

// CSOURCE_BINARY_OP_2D(ft_binary_op_subtract, -, float);

const int thread_count = 8;

struct binary_op_arg {
	float* a, *b, *c;
	unsigned len;
	binary_op_arg(float* a=NULL, float* b=NULL, float*c =NULL, unsigned len=0):
  a(a), b(b), c(c), len(len){}
};

CSOURCE_BINARY_OP(add, +);

CSOURCE_BINARY_OP(multiply, *);

CSOURCE_BINARY_OP(subtract, -)

struct unary_op_arg {
	float* a, *b;
	unsigned len;
	unary_op_arg(float* a=NULL, float* b=NULL, unsigned len=0):
  a(a), b(b), len(len){}
};

void *abs_thread(void *arg0) {               
  unary_op_arg my_arg = *(unary_op_arg*)arg0;                     
  for (unsigned i = 0; i < my_arg.len; ++i)     
    my_arg.b[i] = fabs(my_arg.a[i]);
  return NULL;                       
}   

extern "C" void ft_unary_op_abs(float* a, float* out, int64_t p_SIZ_) {
  unsigned thread, block_size, start;                                               
  pthread_t *thread_handles;                                                        
  unary_op_arg *args;                                                                        
  thread_handles =(pthread_t *)malloc(thread_count * sizeof(pthread_t));            
  args = (unary_op_arg*)malloc(thread_count * sizeof(unary_op_arg));                                  \
  for (thread = 0, start = 0; thread < thread_count; ++thread) {                    
    block_size = p_SIZ_ / thread_count + (thread < p_SIZ_ % thread_count);          
    args[thread] = unary_op_arg(a + start, out + start, block_size);              
    pthread_create(&thread_handles[thread], NULL,                               
    abs_thread, (void *)&args[thread]);                                      
    start += block_size ;                                                           
  }                                                                                 
  for (thread = 0; thread < thread_count; ++thread)                                 
    pthread_join(thread_handles[thread], NULL);                                     
  free(thread_handles);                                                             
  free(args);     
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
