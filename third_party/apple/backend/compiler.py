import functools
from typing import Any, Dict
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget


class AppleBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'apple'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = 'metallib'

    def parse_options(self, opts) -> Any:
        return opts

    def add_stages(self, stages, options, language):
        raise NotImplementedError("compilation pipeline not yet implemented")

    def load_dialects(self, ctx):
        pass

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    @functools.lru_cache()
    def hash(self):
        return 'apple-v0'
