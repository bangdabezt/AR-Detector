import json

# Input and output file paths
input_file = 'test_odvg_exemplars.jsonl'
output_file = 'odvg_test.jsonl'

def reformat_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Load the JSON object from the line
            data = json.loads(line)

            # Retain only the first instance from the detection
            first_instance = data['detection']['instances'][0]
            data['detection']['instances'] = [first_instance]

            # Create the 'query_file' dictionary
            query_file = {
                'filename': data['filename'],
                'height': data['height'],
                'width': data['width'],
                'label': first_instance['label'],
                'category': first_instance['category'],
                'exemplar': [data['exemplars'][0]]
            }

            # Add the new 'query_file' field to the data
            data['query_file'] = query_file

            # Remove the 'exemplars' field from the data
            del data['exemplars']

            # Write the modified data to the output file in JSONL format
            outfile.write(json.dumps(data) + '\n')

# Run the reformatting function
reformat_jsonl(input_file, output_file)
