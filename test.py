import os
import torch

# Replace with the actual path to your .pkl file
file_path = os.path.join('.', 'attribution_eval', 'results-0.pkl')

# Use torch.load to load the data
data = torch.load(file_path)

# Print the data to verify
print(data)

# for i in range(10): (data['res_info'][i][0][2] - data['res_info'][i][0][0])*(data['res_info'][i][0][3] - data['res_info'][i][0][1])
# Print the data to see what it contains
import pdb; pdb.set_trace()
print(data)
