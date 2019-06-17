#!/usr/bin/env python
import multiprocessing.pool
import os

from setuptools import setup, find_packages, distutils
from torch.utils.cpp_extension import BuildExtension

this_file = os.path.dirname(__file__)


# monkey-patch for parallel compilation
# See: https://stackoverflow.com/a/13176803
def parallelCCompile(self,
                     sources,
                     output_dir=None,
                     macros=None,
                     include_dirs=None,
                     debug=0,
                     extra_preargs=None,
                     extra_postargs=None,
                     depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)


    # parallel code
    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    thread_pool = multiprocessing.pool.ThreadPool(4)
    list(thread_pool.imap(_single_compile, objects))
    return objects


# hack compile to support parallel compiling
distutils.ccompiler.CCompiler.compile = parallelCCompile
import build 

setup(
    name="lm_decoder",
    version="0.1",
    description="KenLM decoder for Non-Autoregressive Models.",
    author="Ryan Leary",
    author_email="ryanleary@gmail.com",
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    ext_modules = [build.extension],
    cmdclass={'build_ext': BuildExtension}
)

