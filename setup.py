#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import glob as glob_module
import shutil
import subprocess
import logging
from distutils.errors import DistutilsError
from setuptools import setup, find_packages, Distribution, Command
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PACKAGE_NAME = "ops_multimodal_fusion"
_BASE_VERSION = "1.0.0"
_soc = os.environ.get("SOC", "")
VERSION = f"{_BASE_VERSION}+{_soc}" if _soc else _BASE_VERSION
DESCRIPTION = "PyTorch Ascend C operator extensions"


class CleanCommand(Command):
    """
    usage: python setup.py clean
    """
    description = "Clean build artifacts from the source tree"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @staticmethod
    def _remove_folder(folder, errors):
        if not os.path.exists(folder):
            return
        try:
            shutil.rmtree(folder)
            logging.info(f"Removed folder: {folder}")
        except OSError as e:
            errors.append(f"{folder}: {e}")

    @staticmethod
    def _remove_artifact(file_path, in_pkg, errors):
        parent = os.path.basename(os.path.dirname(file_path))
        is_cache = file_path.endswith(('.pyc', '.pyo')) and parent == '__pycache__'
        is_lib = in_pkg and file_path.endswith(('.so', '.abi3.so'))
        if not (is_cache or is_lib):
            return
        try:
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")
        except OSError as e:
            errors.append(f"{file_path}: {e}")

    def run(self):
        errors = []
        folders_to_remove = ['build', 'dist', f'{PACKAGE_NAME}.egg-info']
        for folder in folders_to_remove:
            self._remove_folder(folder, errors)

        skip_dirs = {'.git', '.hg', '.svn', '.venv', 'venv', 'env', 'node_modules'}
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            in_pkg = root.startswith(f'./{PACKAGE_NAME}')
            for file in files:
                self._remove_artifact(os.path.join(root, file), in_pkg, errors)

        if errors:
            raise DistutilsError("clean failed for:\n  " + "\n  ".join(errors))
        logging.info("Cleaned build artifacts.")


class BinaryDistribution(Distribution):
    """
    Make this wheel not a pure python package
    """
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


class ABI3Wheel(bdist_wheel):
    """
    Force to use actual python version tag for wheel, e.g. cp310-cp310
    """
    def get_tag(self):
        python, abi, plat = super().get_tag()
        python = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi = python
        return python, abi, plat

    def run(self):
        self.run_command('cmake_build')
        super().run()


class CMakeBuildCommand(Command):
    """
    Custom command to build CMake extensions
    """
    description = "Build CMake extensions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cpu_count = os.cpu_count() or 2
        num_jobs = str(cpu_count)
        import torch
        torch_cmake_path = torch.utils.cmake_prefix_path
        torch_dir = os.path.join(torch_cmake_path, "Torch")
        logging.info(f"Using Torch path: {torch_dir}")
        import torch_npu
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        logging.info(f"Using Torch NPU path: {torch_npu_path}")

        npu_arch = os.environ.get('NPU_ARCH', 'dav-3510')
        arch_dir = os.environ.get('ARCH_DIR', 'arch35')
        op_list = os.environ.get('OP_LIST', '')
        logging.info(f"Using NPU_ARCH: {npu_arch}")
        logging.info(f"Using ARCH_DIR: {arch_dir}")
        if op_list:
            logging.info(f"Building ops: {op_list}")

        build_temp = os.path.join(os.getcwd(), 'build')
        cmake_config_command = ['cmake', '-S', os.getcwd(), '-B', build_temp,
                                '-DCMAKE_BUILD_TYPE=Release',
                                f'-DTorch_DIR={torch_dir}',
                                f'-DTORCH_NPU_PATH={torch_npu_path}',
                                f'-DNPU_ARCH={npu_arch}',
                                f'-DARCH_DIR={arch_dir}',
                                f'-DOP_LIST={op_list}'
                                ]
        subprocess.check_call(cmake_config_command, cwd=os.getcwd())

        build_command = ['cmake', '--build', build_temp, '--config', 'Release',
                         '--parallel', num_jobs]
        subprocess.check_call(build_command, cwd=os.getcwd())

        logging.info("CMake extensions built successfully.")


class BuildPyWithSo(build_py):
    """
    Override build_py to copy .so from build/lib/ into the build staging area.
    """
    def run(self):
        self.run_command('cmake_build')
        super().run()
        cmake_lib_dir = os.path.join(os.getcwd(), 'build', 'lib')
        pkg_build_dir = os.path.join(self.build_lib, PACKAGE_NAME)
        os.makedirs(pkg_build_dir, exist_ok=True)
        so_files = glob_module.glob(os.path.join(cmake_lib_dir, 'libops_multimodal_fusion*.so'))
        if not so_files:
            raise RuntimeError(
                f"No libops_multimodal_fusion*.so found in {cmake_lib_dir}; "
                "cmake build may have failed or the output path has changed")
        for so_file in so_files:
            dst = os.path.join(pkg_build_dir, os.path.basename(so_file))
            shutil.copy2(so_file, dst)
            logging.info(f"Copied {os.path.basename(so_file)} to {pkg_build_dir}/")


cmdclass = {
    'clean': CleanCommand,
    'bdist_wheel': ABI3Wheel,
    'cmake_build': CMakeBuildCommand,
    'build_py': BuildPyWithSo,
}


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=[
        "torch",
        "torch_npu"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
)
