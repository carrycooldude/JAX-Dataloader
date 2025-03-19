# JAX DataLoader

A lightweight DataLoader for JAX to load data from various file formats, including CSV, JSON, and more. The goal of this project is to port TensorFlow Dataset (TFDS) functionality into JAX while supporting multiple data sources and preprocessing.

## Features:
- Load data from multiple sources (CSV, JSON, and more).
- Parallel data loading using Python's `multiprocessing`.
- JAX integration for optimized data preprocessing using `vmap`.
- Easy-to-use interface for batch loading.
- JAX-based preprocessing using `jit` and `vmap`.

## Installation

You can install the required dependencies with the following command:

```bash
pip install jax jaxlib pandas numpy
```

### Optional (For multiprocessed data loading):
```bash
pip install multiprocessing
```

## Usage

### 1. **Basic Data Loading from CSV**

This example shows how to load data from a CSV file, specify the target column (label), and use batching with `JAXDataLoader`.

```python
import numpy as np
from jax_dataloader import JAXDataLoader, load_custom_data

# Example 1: Loading CSV data
dataset_path = 'path_to_your_dataset.csv'
batch_size = 32
dataloader = load_custom_data(dataset_path, file_type='csv', batch_size=batch_size, target_column='median_house_value')

for batch_x, batch_y in dataloader:
    print(batch_x.shape, batch_y.shape)
```

### 2. **Data Loading from JSON**

This example shows how to load data from a JSON file.

```python
# Example 2: Loading JSON data
dataset_path = 'path_to_your_dataset.json'
batch_size = 32
dataloader = load_custom_data(dataset_path, file_type='json', batch_size=batch_size, target_column='median_house_value')

for batch_x, batch_y in dataloader:
    print(batch_x.shape, batch_y.shape)
```

### 3. **Load Data from Custom Sources**

You can easily extend the `load_custom_data` function to support additional file formats by adding a custom data loading function and handling it in the `file_type` argument.

```python
# Example 3: Load from a custom source
dataset_path = 'path_to_your_custom_data_file'
file_type = 'your_file_type'  # Can be 'csv', 'json', etc.
batch_size = 64
dataloader = load_custom_data(dataset_path, file_type=file_type, batch_size=batch_size, target_column='your_target_column')
```


## Contributing

Feel free to contribute by submitting issues and pull requests. If you want to add new features or improve the performance, your contributions are welcome!

## License

MIT License. See [LICENSE](LICENSE) for more details.

---

### **Project Structure**:

```
jax-dataloader/
│
├── jax_dataloader.py   # Contains the JAXDataLoader class and data loading logic
├── dataset/            # Example dataset folder
│   ├── housing.csv     # Example CSV data
│   └── housing.json    # Example JSON data
├── README.md           # This README file
└── requirements.txt    # Python dependencies
```

---

### **Pushing to GitHub**:
1. Initialize a Git repository:
    ```bash
    git init
    ```

2. Add your files:
    ```bash
    git add .
    ```

3. Commit your changes:
    ```bash
    git commit -m "Initial commit: JAX DataLoader"
    ```

4. Push to GitHub:
    ```bash
    git remote add origin https://github.com/your-username/jax-dataloader.git
    git push -u origin master
    ```
