"""FT library supported operators.
"""
import logging

import tvm.ir
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr import GlobalVar
from tvm.relay.expr_functor import ExprMutator, ExprVisitor

from ... import _ffi_api
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

logger = logging.getLogger("FT")


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by FT-2500.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by FT.
    """

    @tvm.ir.register_op_attr(op_name, "target.ft")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("subtract")
_register_external_op_helper("add")
_register_external_op_helper("multiply")
_register_external_op_helper("abs")
_register_external_op_helper("nn.relu")


@register_pattern_table("ft")
def pattern_table():
    """Create ft patterns.

    Returns
    -------
    ft_patterns : List[ft_pattern]
        Created patterns.
    """
    ft_patterns = []
    return ft_patterns

