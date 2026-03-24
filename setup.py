from __future__ import annotations

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

compile_args = ["-O3", "-fvisibility=hidden"]
link_args: list[str] = []

if os.name != "nt":
    compile_args.append("-pthread")
    link_args.append("-pthread")

ext_modules = [
    Pybind11Extension(
        "vector_search_cpp",
        ["cpp/vector_search.cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="enterprise-rag-vector-search",
    version="0.1.0",
    description="SIMD + multithreaded vector search extension",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
