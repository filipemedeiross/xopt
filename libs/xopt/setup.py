from pathlib    import Path
from setuptools import setup

from pybind11.setup_helpers import build_ext, \
                                   Pybind11Extension


ROOT    = Path(__file__).resolve().parents[2]
SRC     = ROOT / "src"
BINDING = SRC  / "python" / "module.cpp"

COMMON = [
    path
    for path in SRC.rglob("*.cpp")
    if  path.name != "main.cpp" and "python" not in path.parts
]

ext_modules = [
    Pybind11Extension(
        "xopt",
        sources     =[str(BINDING), *map(str, COMMON)],
        include_dirs=[str(SRC    )],
        cxx_std=20,
    )
]

setup(
    name   ="xopt" ,
    version="0.1.0",
    description="Python bindings for the XOpt tabu search and k-medoids toolkit.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)