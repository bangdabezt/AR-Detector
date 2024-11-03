import json
import random
from collections import defaultdict

def split_by_label_and_filename(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    with open(input_file, 'r') as f:
        records = [json.loads(line) for line in f]

    # Group records by `label` and then by `filename` within each label
    label_filename_groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        label = record['detection']['instances'][0]['label']
        filename = record['filename']
        label_filename_groups[label][filename].append(record)

    # Prepare lists to store train, validation, and test records
    train_records, val_records, test_records = [], [], []

    # Process each label independently
    for label, filename_groups in label_filename_groups.items():
        filenames = list(filename_groups.keys())
        random.shuffle(filenames)

        # Calculate split indices
        num_train = int(len(filenames) * train_ratio)
        num_val = int(len(filenames) * val_ratio)

        train_filenames = filenames[:num_train]
        val_filenames = filenames[num_train:num_train + num_val]
        test_filenames = filenames[num_train + num_val:]

        # Assign records to train, validation, and test based on filename split
        for filename in train_filenames:
            train_records.extend(filename_groups[filename])
        for filename in val_filenames:
            val_records.extend(filename_groups[filename])
        for filename in test_filenames:
            test_records.extend(filename_groups[filename])

    # Write each partition to its respective file
    with open(train_file, 'w') as train_f:
        for record in train_records:
            train_f.write(json.dumps(record) + '\n')
    
    with open(val_file, 'w') as val_f:
        for record in val_records:
            val_f.write(json.dumps(record) + '\n')
    
    with open(test_file, 'w') as test_f:
        for record in test_records:
            test_f.write(json.dumps(record) + '\n')

# Example usage
split_by_label_and_filename(
    input_file='output.jsonl',
    train_file='train_lafi.jsonl',
    val_file='val_lafi.jsonl',
    test_file='test_lafi.jsonl',
    train_ratio=0.8,  # 80% of records for training
    val_ratio=0.1,    # 10% for validation
    test_ratio=0.1,   # 10% for testing
    seed=42           # Set seed for consistency
)
