import numpy as np
import pandas as pd
import json
from PIL import Image
import os
from jax_dataloader import JAXDataLoader, DataLoaderConfig, load_custom_data

def generate_test_data():
    """Generate test data in different formats"""
    # 1. Generate CSV data (simulated sensor readings)
    print("\nGenerating CSV data...")
    sensor_data = {
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 10, 1000),
        'pressure': np.random.normal(1013, 10, 1000),
        'status': np.random.randint(0, 4, 1000)  # 4 different status classes
    }
    df = pd.DataFrame(sensor_data)
    df.to_csv('test_data/structured/sensor_readings.csv', index=False)
    print("CSV data saved to test_data/structured/sensor_readings.csv")

    # 2. Generate JSON data (simulated user behavior)
    print("\nGenerating JSON data...")
    user_data = []
    for _ in range(1000):
        features = [
            np.random.randint(1, 100),  # clicks
            np.random.random() * 60,     # time spent
            np.random.randint(1, 5),     # pages visited
            np.random.random()           # conversion score
        ]
        label = 1 if np.mean(features) > np.median(features) else 0  # engagement label
        user_data.append({'features': features, 'label': label})
    
    with open('test_data/structured/user_behavior.json', 'w') as f:
        json.dump(user_data, f)
    print("JSON data saved to test_data/structured/user_behavior.json")

    # 3. Generate Image data (simple shapes)
    print("\nGenerating Image data...")
    os.makedirs('test_data/images', exist_ok=True)
    shapes = ['circle', 'square', 'triangle']
    
    for i in range(100):
        shape_type = i % len(shapes)
        img = Image.new('RGB', (64, 64), 'white')
        pixels = np.array(img)
        
        # Draw different shapes
        if shapes[shape_type] == 'circle':
            xx, yy = np.mgrid[:64, :64]
            circle = (xx - 32) ** 2 + (yy - 32) ** 2 <= 25 ** 2
            pixels[circle] = [255, 0, 0]  # Red circle
        elif shapes[shape_type] == 'square':
            pixels[20:44, 20:44] = [0, 255, 0]  # Green square
        else:
            xx, yy = np.mgrid[:64, :64]
            triangle = (xx + yy <= 80) & (xx >= 20) & (yy >= 20)
            pixels[triangle] = [0, 0, 255]  # Blue triangle
            
        img = Image.fromarray(pixels)
        img.save(f'test_data/images/{shape_type}_{i}.png')
    print("Image data saved to test_data/images/")

def test_csv_dataloader():
    """Test DataLoader with CSV data"""
    print("\nTesting CSV DataLoader...")
    dataloader = load_custom_data(
        'test_data/structured/sensor_readings.csv',
        file_type='csv',
        batch_size=32,
        target_column='status'
    )
    
    print("DataLoader configuration:")
    print(dataloader.get_stats())
    
    print("\nLoading first batch...")
    batch_x, batch_y = next(iter(dataloader))
    print(f"Batch features shape: {batch_x.shape}")
    print(f"Batch labels shape: {batch_y.shape}")

def test_json_dataloader():
    """Test DataLoader with JSON data"""
    print("\nTesting JSON DataLoader...")
    dataloader = load_custom_data(
        'test_data/structured/user_behavior.json',
        file_type='json',
        batch_size=32,
        multi_gpu=True
    )
    
    print("DataLoader configuration:")
    print(dataloader.get_stats())
    
    print("\nLoading first batch...")
    batch_x, batch_y = next(iter(dataloader))
    print(f"Batch features shape: {batch_x.shape}")
    print(f"Batch labels shape: {batch_y.shape}")

def test_image_dataloader():
    """Test DataLoader with Image data"""
    print("\nTesting Image DataLoader...")
    dataloader = load_custom_data(
        'test_data/images',
        file_type='image',
        batch_size=16,
        pinned_memory=True
    )
    
    print("DataLoader configuration:")
    print(dataloader.get_stats())
    
    print("\nLoading first batch...")
    batch_x, batch_y = next(iter(dataloader))
    print(f"Batch images shape: {batch_x.shape}")
    print(f"Batch labels shape: {batch_y.shape}")

if __name__ == "__main__":
    # Generate test data
    generate_test_data()
    
    # Test DataLoader with different data types
    test_csv_dataloader()
    test_json_dataloader()
    test_image_dataloader() 