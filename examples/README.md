# JAX DataLoader Examples

This directory contains examples demonstrating various features of the JAX DataLoader package.

## Setup

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Examples

### Data Loading Demo (`data_loading/`)
Demonstrates loading and processing different types of data:
- CSV files (structured tabular data)
- JSON files (nested structured data)
- Image files (computer vision data)

Features demonstrated:
- Multi-GPU support
- Memory management
- Automatic batch size tuning
- Progress tracking
- Different data format handling

To run the demo:
```bash
cd data_loading
python demo.py
```

The demo will:
1. Generate sample data in different formats
2. Load and process the data using JAX DataLoader
3. Display statistics and batch information

## Example Output
```
Generating CSV data...
CSV data saved to test_data/structured/sensor_readings.csv

Generating JSON data...
JSON data saved to test_data/structured/user_behavior.json

Generating Image data...
Image data saved to test_data/images/

Testing CSV DataLoader...
DataLoader configuration:
{'total_samples': 1000, 'batch_size': 32, ...}

Loading first batch...
Batch features shape: (32, 3)
Batch labels shape: (32,)
...
```

## Custom Usage
You can modify the examples to suit your needs:
- Adjust batch sizes
- Enable/disable features like multi-GPU support
- Change data formats or structures
- Modify data generation parameters

## Contributing
Feel free to contribute additional examples by submitting a pull request! 