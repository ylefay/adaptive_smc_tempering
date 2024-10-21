import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="smc",
    author="Yvann Le Fay",
    description="SMC tempering.",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "pytest",
        "scipy",
        "blackjax",
        "jax",
        "jaxlib",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "particles"
    ],
    long_description_content_type="text/markdown",
    keywords="smc tempering gaussian",
    license="MIT",
    license_files=("LICENSE",),
)
