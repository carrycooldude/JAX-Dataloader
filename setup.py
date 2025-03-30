"""Setup configuration for JAX DataLoader."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jax-dataloaders",
    version="0.1.7",
    author="Kartikey Rawat",
    author_email="rawatkari554@gmail.com",
    description="A high-performance data loading library for JAX applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carrycooldude/JAX-Dataloader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "Pillow>=8.0.0",
        "psutil>=5.8.0",
        "tqdm>=4.50.0",
    ],
    include_package_data=True,
    package_data={
        "jax_dataloader": ["py.typed"],
        "examples": ["requirements.txt", "README.md"],
    },
    data_files=[
        ("examples", ["examples/requirements.txt", "examples/README.md"]),
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "docutils>=0.16",
            "build>=0.7.0",
            "twine>=3.4.2",
        ],
        "csv": ["pandas>=2.0.0"],
        "json": ["pandas>=2.0.0"],
        "image": ["pillow>=10.0.0"],
        "all": ["pandas>=2.0.0", "pillow>=10.0.0"],
    },
)
