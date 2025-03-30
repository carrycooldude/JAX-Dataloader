from setuptools import setup, find_packages

setup(
    name='jax_dataloaders',  # Your package name
    version='0.1.0',  # Version number, increment this for each release
    packages=find_packages(),  # This automatically finds all the packages in your project
    install_requires=[
        'sphinx',  # List any dependencies required for your project
        'sphinx_rtd_theme',
    ],
    package_data={
        '': ['docs/*', 'requirements.txt'],  # Include docs and any other files in the package
    },
    classifiers=[
        'Programming Language :: Python :: 3',  # Specify which Python versions your package supports
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version
    long_description=open('README.md').read(),  # This will add the contents of your README to the PyPI page
    long_description_content_type='text/markdown',  # To specify Markdown format for the README
    author='Kartikey Rawat', 
    author_email='rawatkari554@gmail.com', 
)
