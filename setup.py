"""Setup configuration for JAX DataLoader."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jax-dataloaders",
    version="0.1.10",
    author="Kartikey Rawat",
    author_email="rawatkari554@gmail.com",
    description="A high-performance data loading library for JAX",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "typing-extensions>=4.5.0",
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
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.0.0",
        ],
        "csv": ["pandas>=2.0.0"],
        "json": ["pandas>=2.0.0"],
        "image": ["pillow>=10.0.0"],
        "all": ["pandas>=2.0.0", "pillow>=10.0.0"],
    },
    enable_jekyll=False,
    disable_nojekyll=False,
)
