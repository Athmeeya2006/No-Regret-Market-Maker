"""
pip install -e .       # Build in place (development mode)
pip install .          # Full install
"""
import subprocess
import sys
import os
from pathlib import Path
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """Delegate extension build to CMake."""

    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        source_dir = Path(__file__).parent.resolve()

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake = shutil.which("cmake")
        venv_cmake = Path(sys.executable).with_name("cmake")
        if venv_cmake.exists():
            cmake = str(venv_cmake)
        if cmake is None:
            raise RuntimeError(
                "CMake is required to build lob_engine. Install it with "
                "`python -m pip install cmake` or your system package manager."
            )

        pybind11_cmake_dir = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            text=True,
        ).strip()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPYBIND11_CMAKE_DIR={pybind11_cmake_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["--config", "Release", "--parallel"]

        subprocess.check_call(
            [cmake, str(source_dir)] + cmake_args,
            cwd=build_temp,
        )
        subprocess.check_call(
            [cmake, "--build", "."] + build_args,
            cwd=build_temp,
        )


setup(
    name="lob_engine",
    version="0.1.0",
    description="C++ Limit Order Book engine with Python bindings",
    ext_modules=[Extension("lob_engine", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
    python_requires=">=3.9",
    install_requires=["pybind11>=2.11", "numpy>=1.24"],
    zip_safe=False,
)
