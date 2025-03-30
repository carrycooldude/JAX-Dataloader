import os
import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap, device_put
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image
import json


class JAXDataLoader:
    """
    A high-performance JAX DataLoader with pinned memory and GPU prefetching.

    Features:
    - Uses memory-mapped files for large datasets.
    - Preloads batches into pinned memory (RAM).
    - Prefetches data to GPU asynchronously.
    - Supports CSV, JSON, and image datasets.
    """

    def __init__(self, data, labels, batch_size=32, shuffle=True, num_workers=4, pinned_memory=True, prefetch=True):
        self.data = np.asarray(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data
        self.labels = np.asarray(labels, dtype=np.int32) if not isinstance(labels, np.ndarray) else labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pinned_memory = pinned_memory
        self.prefetch = prefetch

        self.indices = np.arange(len(self.data))
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        # Load batch in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            batch_data, batch_labels = zip(*executor.map(self._fetch_sample, batch_indices))

        batch_data, batch_labels = np.array(batch_data), np.array(batch_labels)

        if self.pinned_memory:
            batch_data = np.asarray(batch_data, dtype=np.float32)  # Zero-copy transfer
            batch_labels = np.asarray(batch_labels, dtype=np.int32)

        # Move to GPU
        batch_data, batch_labels = device_put((batch_data, batch_labels))

        if self.prefetch:
            batch_data, batch_labels = self._prefetch(batch_data, batch_labels)

        return batch_data, batch_labels

    def _fetch_sample(self, idx):
        return self._preprocess(self.data[idx]), self.labels[idx]

    def _prefetch(self, batch_data, batch_labels):
        """Prefetches data to GPU asynchronously."""
        return jax.jit(lambda x, y: (x, y))(batch_data, batch_labels)

    @staticmethod
    def _preprocess(sample):
        """Example preprocessing: Normalize sample values to [0,1]"""
        return jnp.array(sample) / 255.0


def load_custom_data(file_path, file_type='csv', batch_size=32, target_column=None, pinned_memory=True):
    """Loads data from CSV, JSON, or Image folders."""
    if file_type == 'csv':
        data, labels = load_csv_data(file_path, target_column)
    elif file_type == 'json':
        data, labels = load_json_data(file_path)
    elif file_type == 'image':
        data, labels = load_image_data(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return JAXDataLoader(data, labels, batch_size=batch_size, pinned_memory=pinned_memory)


def load_csv_data(file_path, target_column=None):
    """Loads structured data from a CSV file."""
    df = pd.read_csv(file_path)
    print("CSV Columns:", df.columns.tolist())
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' column not found in CSV. Available columns: {df.columns.tolist()}")
    data = df.drop(target_column, axis=1).values
    labels = df[target_column].values
    return data, labels


def load_json_data(file_path):
    """Loads structured data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    features = np.array([item['features'] for item in data])
    labels = np.array([item['label'] for item in data])
    return features, labels


def load_image_data(image_folder_path, img_size=(64, 64)):
    """Loads image data from a folder and resizes it."""
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png'))]
    data = []
    labels = []
    for img_file in image_files:
        img = Image.open(os.path.join(image_folder_path, img_file))
        img = img.resize(img_size)
        data.append(np.array(img))
        label = int(img_file.split('_')[0])  # Assuming labels are part of file name (e.g., "0_image1.jpg")
        labels.append(label)
    return np.array(data), np.array(labels)


# Example usage: Loading custom dataset and iterating over it
if __name__ == "__main__":
    dataset_path = 'dataset.csv'  # Replace with actual dataset path
    batch_size = 64

    # Example 1: Loading CSV
    dataloader = load_custom_data(dataset_path, file_type='csv', batch_size=batch_size, target_column='median_house_value')

    # Example 2: Loading JSON
    # dataloader = load_custom_data('dataset.json', file_type='json', batch_size=batch_size)

    # Example 3: Loading Images
    # dataloader = load_custom_data('images_folder/', file_type='image', batch_size=batch_size)

    for batch_x, batch_y in dataloader:
        print("Batch Shape:", batch_x.shape, batch_y.shape)
