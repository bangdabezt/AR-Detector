import random
import json
# Set the seed for reproducibility
random.seed(42)

# Read all lines from the original JSONL file
original_lines = []
with open('output.jsonl', 'r') as file:
    original_lines = file.readlines()

# Shuffle the list randomly
random.shuffle(original_lines)

# Total number of lines
total_lines = len(original_lines)

# Define proportions for training, validation, and test sets
train_ratio = 0.8
val_ratio = 0.1

# Compute split indices
train_index = int(train_ratio * total_lines)
val_index = train_index + int(val_ratio * total_lines)

# Split the lines
train_lines = original_lines[:train_index]
val_lines = original_lines[train_index:val_index]
test_lines = original_lines[val_index:]

# Write the splits to new JSONL files
with open('train_random.jsonl', 'w') as file:
    file.writelines(train_lines)

with open('val_random.jsonl', 'w') as file:
    file.writelines(val_lines)

with open('test_random.jsonl', 'w') as file:
    file.writelines(test_lines)

category_mapping = {}
label_map = {}
for line in original_lines:
    data = json.loads(line.strip())["detection"]["instances"][0]
    category_name = data["category"]
    category_id = data["label"]
    if category_name not in category_mapping:
        category_mapping[category_name] = category_id
        label_map[str(category_id)] = category_name

with open('label_map.json', 'w') as json_file:
    json.dump(label_map, json_file)
# import pdb; pdb.set_trace()
