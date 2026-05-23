import functools
import subprocess
import tempfile
import os
import re
from pathlib import Path
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

    def load_dialects(self, ctx):
        pass

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    @functools.lru_cache()
    def hash(self):
        return 'apple-v0'

    # ---- Compilation stages ----

    def make_llir(self, src, metadata, options):
        """Lower TTGIR to LLVM IR (MLIR), then convert to LLVM IR (LLVM)."""
        from triton._C.libtriton import ir, passes, llvm

        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.ttgpuir.add_to_llvmir(pm, 0, 0, False)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod, 'make_llir')

        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        llvm.attach_datalayout(llvm_mod, 'spir64-unknown-unknown', '', '')
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        metadata["shared"] = src.get_int_attr("ttg.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def _llvm_to_spirv(self, src: str, metadata: dict) -> bytes:
        """Convert LLVM IR assembly to SPIRV binary via llvm-spirv."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.ll', delete=False
        ) as ll_file:
            ll_file.write(src)
            ll_path = ll_file.name

        spirv_path = ll_path + '.spv'
        try:
            subprocess.run(
                ['llvm-spirv', ll_path, '-o', spirv_path],
                check=True, capture_output=True, text=True
            )
            with open(spirv_path, 'rb') as f:
                return f.read()
        finally:
            for p in [ll_path, spirv_path]:
                if os.path.exists(p):
                    os.remove(p)

    def _spirv_to_msl(self, src: bytes, metadata: dict) -> str:
        """Convert SPIRV binary to Metal Shading Language via spirv-cross."""
        with tempfile.NamedTemporaryFile(
            suffix='.spv', delete=False
        ) as spv_file:
            spv_file.write(src)
            spv_path = spv_file.name

        try:
            result = subprocess.run(
                ['spirv-cross', spv_path, '--msl',
                 '--msl-version', '30000'],
                check=True, capture_output=True, text=True
            )
            return result.stdout
        finally:
            if os.path.exists(spv_path):
                os.remove(spv_path)

    def _msl_to_metallib(self, src: str, metadata: dict) -> bytes:
        """Compile MSL source to Metal library via Metal compiler."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.metal', delete=False
        ) as msl_file:
            msl_file.write(src)
            msl_path = msl_file.name

        metallib_path = msl_path + '.metallib'
        try:
            sdk_path = subprocess.check_output(
                ['xcrun', '--sdk', 'macosx', '--show-sdk-path'],
                text=True
            ).strip()

            subprocess.run(
                ['xcrun', '-sdk', 'macosx', 'metal',
                 '-c', msl_path,
                 '-o', metallib_path,
                 '-isysroot', sdk_path,
                 '-std=metal3.0'],
                check=True, capture_output=True, text=True
            )
            with open(metallib_path, 'rb') as f:
                return f.read()
        finally:
            for p in [msl_path, metallib_path]:
                if os.path.exists(p):
                    os.remove(p)

    def add_stages(self, stages, options, language):
        """Wire the compilation pipeline stages."""
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["spirv"] = lambda src, metadata: self._llvm_to_spirv(src, metadata)
        stages["msl"] = lambda src, metadata: self._spirv_to_msl(src, metadata)
        stages["metallib"] = lambda src, metadata: self._msl_to_metallib(src, metadata)
