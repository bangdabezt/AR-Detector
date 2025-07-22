import json
import random
from collections import defaultdict

def partition_by_label_with_categories(input_file, train_file, val_file, test_file, val_categories, test_categories, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    with open(input_file, 'r') as f:
        records = [json.loads(line) for line in f]

    # Group records by `label` (we assume that label is under `detection->instances[0]->label`)
    label_groups = defaultdict(list)
    label_to_category = {}  # Map labels to categories for easy lookup
    for record in records:
        label = record['detection']['instances'][0]['label']
        category = record['detection']['instances'][0]['category']
        label_groups[label].append(record)
        label_to_category[label] = category

    # Determine the labels for validation and test sets based on specified categories
    val_labels = [label for label, cat in label_to_category.items() if cat in val_categories]
    test_labels = [label for label, cat in label_to_category.items() if cat in test_categories]

    # Prepare lists to store train, validation, and test records
    train_records, val_records, test_records = [], [], []

    # Distribute records based on the specified labels
    for label, records_for_label in label_groups.items():
        if label in val_labels:
            val_records.extend(records_for_label)
        elif label in test_labels:
            test_records.extend(records_for_label)
        else:
            train_records.extend(records_for_label)

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
partition_by_label_with_categories(
    input_file='output.jsonl',
    train_file='train_label2.jsonl',
    val_file='val_teddy.jsonl',
    test_file='test_sheep.jsonl',
    val_categories=['teddy bear'],  # Specify categories for validation
    test_categories=['sheep'],   # Specify categories for testing
    seed=42                      # Set seed for consistency
)

