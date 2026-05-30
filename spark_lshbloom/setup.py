from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf8")


setup(
    name="spark-lshbloom",
    version="0.1.0",
    description="Spark-based LSHBloom text deduplication with Bloom and banding modes.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="LSHBloom Spark Deduplication Contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9,<3.13",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "pyarrow>=10",
        "pyspark>=3.4",
        "psutil>=5.9",
        "xxhash>=3.4",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
