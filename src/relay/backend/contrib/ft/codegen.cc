/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief Extended from example codegen for FT-2500.
 */
class CodegenFT : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit CodegenFT(const std::string& id) { this->ext_func_id_ = id; }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "FT codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const TupleNode* node) final {
    std::vector<Output> outs;
    for (auto field : node->fields) {
      auto res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
      outs.push_back(res[0]);
    }
    return outs;
  }

  std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
    auto res = VisitExpr(op->tuple);
    ICHECK_GT(res.size(), static_cast<size_t>(op->index));

    // Only keep the item we want for the child node.
    // FIXME(@comaniac): The other items should still be requried for the primary outputs.
    return {res[op->index]};
  }

  std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    Output output;
    // Get const: static_cast<float*>(gcc_0_consts[0]->data)
    output.name = CreateDataReference(ext_func_id_, const_idx_);
    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);

    // Generate the global variable for needed ndarrays
    if (const_array_name_.empty()) {
      const_array_name_ = CreateNDArrayPool(ext_func_id_);
      std::string checker = CreateInitChecker(ext_func_id_);
      ext_func_body_.insert(ext_func_body_.begin(), checker);
    }

    ICHECK(dtype == "float" || dtype == "int") << "Only float and int are supported for now.";
    output.dtype = dtype;

    std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
    const_vars_.push_back(const_var_name);
    const_idx_++;

    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

    const auto* type_node = call->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);

    // Make function declaration
    
    if (IsOp(call, "nn.conv2d")) {
      decl_stream << "ft_conv2d(";

      // Make function call when visiting arguments
      bool first = true;
      for (size_t i = 0; i < call->args.size(); ++i) {
        auto res = VisitExpr(call->args[i]);
        for (auto out : res) {
          if (!first) {
            decl_stream << ", ";
          }
          first = false;
          decl_stream << out.name;
        }
      }

      std::string out = "buf_" + std::to_string(buf_idx_++);
      auto out_shape = GetShape(call->checked_type());
      int out_size = 1;
      for (size_t i = 0; i < out_shape.size(); ++i) {
        out_size *= out_shape[i];
      }
      buf_stream << dtype << "* " << out << " = (" << dtype << "*)malloc(4 * " << out_size << ");";
      buf_decl_.push_back(buf_stream.str());

      decl_stream << ", " << out << ", std::vector<int>({";
      auto data_shape = GetShape(call->args[0]->checked_type());
      for (size_t i = 0, first = true; i < data_shape.size(); ++i) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << data_shape[i];
      }
      decl_stream << "}), std::vector<int>({";
      auto weight_shape = GetShape(call->args[1]->checked_type());
      for (size_t i = 0, first = true; i < weight_shape.size(); ++i) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << weight_shape[i];
      }
      decl_stream << "}));";
      ext_func_body_.push_back(decl_stream.str());

      // Update output buffer
      // Note C codegen only handles TensorType. Therefore, we don't flatten
      // tuples and only return a single vaule.
      Output output;
      output.name = out;
      output.dtype = dtype;
      output.need_copy = true;
      output.size = out_size;
      return {output};
    } else {
      const auto* op_node = call->op.as<OpNode>();
      ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();

      static const std::map<std::string, std::string> op_map = {
          {"add", "ft_binary_op_add"},           {"subtract", "ft_binary_op_subtract"},
          {"multiply", "ft_binary_op_multiply"}, {"abs", "ft_unary_op_abs"},
          {"nn.relu", "ft_unary_op_relu"},       {"sigmod", "ft_unary_op_sigmod"},
          {"tanh", "ft_unary_op_tanh"},
      };

      const auto op_name = GetRef<Op>(op_node)->name;
      const auto iter = op_map.find(op_name);
      if (iter == op_map.end()) {
        LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
        return {};
      }

      decl_stream << iter->second << "(";

      // Make function call when visiting arguments
      bool first = true;
      for (size_t i = 0; i < call->args.size(); ++i) {
        auto res = VisitExpr(call->args[i]);
        for (auto out : res) {
          if (!first) {
            decl_stream << ", ";
          }
          first = false;
          decl_stream << out.name;
        }
      }

      std::string out = "buf_" + std::to_string(buf_idx_++);
      auto out_shape = GetShape(call->checked_type());
      int out_size = 1;
      for (size_t i = 0; i < out_shape.size(); ++i) {
        out_size *= out_shape[i];
      }
      buf_stream << dtype << "* " << out << " = (" << dtype << "*)malloc(4 * " << out_size << ");";
      buf_decl_.push_back(buf_stream.str());

      decl_stream << ", " << out ;
      auto in_shape = GetShape(call->args[0]->checked_type());
      int in_size = 1;
      for (size_t i = 0; i < in_shape.size(); ++i) {
        in_size *= in_shape[i];
      }
      decl_stream << ", " << in_size << ");";
      ext_func_body_.push_back(decl_stream.str());

      // Update output buffer
      // Note C codegen only handles TensorType. Therefore, we don't flatten
      // tuples and only return a single vaule.
      Output output;
      output.name = out;
      output.dtype = dtype;
      output.need_copy = true;
      output.size = out_size;
      return {output};
    }
  }

  /*!
   * \brief Emit the source code that invokes C compiler compatible wrappers.
   *
   * \return The emitted code.
   */
  std::string JIT(const std::vector<Output>& out) {
    // Write function macros
    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

 private:
  /*! \brief The function id that represents a C source function. */
  std::string ext_func_id_ = "";
  /*! \brief The index of a wrapped C function. */
  int func_idx = 0;
  /*! \brief The index of allocated buffers. */
  int buf_idx_ = 0;
  /*! \brief The index of global constants. */
  int const_idx_ = 0;
  /*! \brief The arguments of a C compiler compatible function. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of a C compiler compatible function. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration statements of a C compiler compatible function. */
  std::vector<std::string> func_decl_;
  /*! \brief The declaration statements of buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The variable name to constant mapping. */
  Array<String> const_vars_;

  friend class FTModuleCodegen;
};

class FTModuleCodegen : public CSourceModuleCodegenBase {
 public:
  std::tuple<Array<String>, String, String> GenCFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";
    CodegenFT builder(GetExtSymbol(func));
    auto out = builder.VisitExpr(func->body);
    return std::make_tuple(builder.const_vars_, builder.ext_func_id_, builder.JIT(out));
  }

  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    ICHECK(ref->IsInstance<FunctionNode>());
    auto res = GenCFunc(Downcast<Function>(ref));
    Array<String> variables = std::get<0>(res);
    String func_name = std::get<1>(res);

    // Create headers
    code_stream_ << "#include <stdio.h>\n";
    code_stream_ << "#include <stdlib.h>\n";
    code_stream_ << "#include <string.h>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/c_backend_api.h>\n";
    code_stream_ << "#include <ft/ft.h>\n";
    code_stream_ << "using namespace tvm::runtime;\n";
    code_stream_ << "using namespace tvm::runtime::contrib;\n";
    code_stream_ << "\n";
    if (!variables.empty()) {
      // This segment would be generated in C++ because of the usage
      // of tvm::runtime::Array. This is not ideal, but this to demonstrate
      // constant copying process used packed imports in other external
      // codegen. Moreover, in microTVM we dont expect this part to be generated.
      code_stream_ << "#ifdef __cplusplus\n";
      code_stream_ << "#include <tvm/runtime/ndarray.h>\n";
      code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
      code_stream_ << "#endif\n";
    }
    code_stream_ << std::get<2>(res);
    std::string code = code_stream_.str();
    FILE *fp = NULL;
    fp = fopen("/home/huangyunyi/codegen_code.cc", "w");
    fprintf(fp, "%s", code.c_str());

    // Create a C Source module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find C Source module to create the external runtime module";
    return (*pf)(code, "c", Array<String>{func_name}, variables);
  }

 private:
  std::ostringstream code_stream_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM, so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 */
runtime::Module FTCompiler(const ObjectRef& ref) {
  FTModuleCodegen csource;
  return csource.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.ft").set_body_typed(FTCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
