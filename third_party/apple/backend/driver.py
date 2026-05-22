import ctypes
import ctypes.util

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _metal_framework_is_available():
    """Check if Metal.framework can be loaded on this system."""
    if not hasattr(ctypes.util, 'find_library'):
        return False
    path = ctypes.util.find_library('Metal')
    if path is None:
        return False
    try:
        lib = ctypes.CDLL(path)
        return hasattr(lib, 'MTLCreateSystemDefaultDevice')
    except OSError:
        return False


class AppleDriver(DriverBase):

    def __init__(self):
        pass

    @classmethod
    def is_active(self):
        return _metal_framework_is_available()

    def get_current_target(self):
        return GPUTarget("apple", "apple-latest", 32)

    def get_active_torch_device(self):
        import torch
        return torch.device("mps")

    def map_python_to_cpp_type(self, ty: str) -> str:
        type_map = {
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "double",
            "bf16": "double",
            "fp32": "double",
            "f32": "double",
            "fp64": "double",
        }
        if ty[0] == '*':
            return "uintptr_t"
        if ty in type_map:
            return type_map[ty]
        raise ValueError(f"Unknown type: {ty}")

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def allocate_default_profile_scratch(self, size: int, alignment: int, stream):
        raise NotImplementedError("not yet implemented")
