import os
import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap
import random
import pandas as pd
from PIL import Image

class JAXDataLoader:
    def __init__(self, data, labels, batch_size=32, shuffle=True, num_workers=4):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)  # Avoid redundant conversion
        self.labels = labels if isinstance(labels, np.ndarray) else np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = np.arange(len(data))  # Track indices for batching
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0  # Reset for new epoch
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data, batch_labels = self.data[batch_indices], self.labels[batch_indices]
        self.current_index += self.batch_size
        return self._parallel_process(batch_data, batch_labels)

    def _parallel_process(self, batch_data, batch_labels):
        processed_data = vmap(self._preprocess)(batch_data)  # Vectorized preprocessing
        return jnp.array(processed_data), jnp.array(batch_labels, dtype=jnp.int32)

    @staticmethod
    def _preprocess(sample):
        # Example preprocessing: Normalize sample values to [0,1]
        return jnp.array(sample) / 255.0  # Modify this for your dataset

def load_custom_data(file_path, file_type='csv', batch_size=32, target_column=None):
    if file_type == 'csv':
        data, labels = load_csv_data(file_path, target_column)
    elif file_type == 'json':
        data, labels = load_json_data(file_path)
    elif file_type == 'image':
        data, labels = load_image_data(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return JAXDataLoader(data, labels, batch_size=batch_size)

def load_csv_data(file_path, target_column=None):
    # Load CSV data using pandas
    df = pd.read_csv(file_path)
    
    # Print column names for debugging
    print("CSV Columns:", df.columns.tolist())
    
    # Check if target_column exists in the dataframe
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' column not found in CSV. Available columns: {df.columns.tolist()}")
    
    data = df.drop(target_column, axis=1).values  # Assuming target_column is the label column
    labels = df[target_column].values
    return data, labels


def load_json_data(file_path):
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Assuming JSON data structure like: [{"features": [..], "label": ..}, ...]
    features = np.array([item['features'] for item in data])
    labels = np.array([item['label'] for item in data])
    return features, labels

def load_image_data(image_folder_path, img_size=(64, 64)):
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg') or f.endswith('.png')]
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
    dataset_path = 'dataset_path'  # Can be CSV, JSON, or image folder path
    batch_size = 64  # Example batch size
    
    # Example 1: Loading CSV
    dataloader = load_custom_data(dataset_path, file_type='csv', batch_size=batch_size, target_column='median_house_value')    
    # Example 2: Loading JSON
    # dataloader = load_custom_data('dataset.json', file_type='json', batch_size=batch_size)

    # Example 3: Loading Images
    # dataloader = load_custom_data('images_folder/', file_type='image', batch_size=batch_size)

    for batch_x, batch_y in dataloader:
        print(batch_x.shape, batch_y.shape)
