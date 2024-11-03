import json
import random
import numpy as np  # Optional if you're using numpy in your code

def partition_by_filename(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)  # Optional if you use numpy

    with open(input_file, 'r') as f:
        records = [json.loads(line) for line in f]

    # Group records by `filename`
    filename_groups = {}
    for record in records:
        filename = record['filename']
        if filename not in filename_groups:
            filename_groups[filename] = []
        filename_groups[filename].append(record)

    # Get all unique filenames and shuffle them for random splitting
    filenames = list(filename_groups.keys())
    random.shuffle(filenames)  # The order will be consistent each time this function is run with the same seed

    # Calculate split indices
    num_train = int(len(filenames) * train_ratio)
    num_val = int(len(filenames) * val_ratio)

    train_filenames = filenames[:num_train]
    val_filenames = filenames[num_train:num_train + num_val]
    test_filenames = filenames[num_train + num_val:]

    # Write each partition to its respective file
    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f, open(test_file, 'w') as test_f:
        for filename in train_filenames:
            for record in filename_groups[filename]:
                train_f.write(json.dumps(record) + '\n')
        for filename in val_filenames:
            for record in filename_groups[filename]:
                val_f.write(json.dumps(record) + '\n')
        for filename in test_filenames:
            for record in filename_groups[filename]:
                test_f.write(json.dumps(record) + '\n')

# Example usage
partition_by_filename(
    input_file='output.jsonl',
    train_file='train_filename.jsonl',
    val_file='val_filename.jsonl',
    test_file='test_filename.jsonl',
    seed=42  # Set a seed value for consistency
)
