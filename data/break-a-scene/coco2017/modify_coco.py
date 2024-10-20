import json
from collections import defaultdict

# Input and output paths
input_file = 'val_coco_exemplars.json'  # Change to your file path
output_file = 'coco_exe.json'

def reformat_coco_data(input_file, output_file):
    # Load the input COCO data
    with open(input_file, 'r') as infile:
        coco_data = json.load(infile)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    # Group annotations by image_id using defaultdict
    grouped_annotations = defaultdict(list)
    for annotation in annotations:
        grouped_annotations[annotation["image_id"]].append(annotation)

    # Prepare new annotations list
    new_annotations = []
    annotation_id = 1

    # Iterate over the images and process their annotations
    for image in images:
        image_id = image["id"]
        if len(grouped_annotations[image_id]) < 2:
            continue  # Skip if fewer than 2 annotations

        # Get the first two annotations
        annotation_1, annotation_2 = grouped_annotations[image_id][:2]

        # Create a merged annotation
        merged_annotation = {
            "id": annotation_id,
            "image_id": annotation_1["image_id"],
            "category_id": annotation_1["category_id"],
            "bbox": annotation_1["bbox"],
            "area": annotation_1["area"],
            "iscrowd": annotation_1["iscrowd"],
            "query_file": {
                "image_id": annotation_2["image_id"],
                "category_id": annotation_2["category_id"],
                "bbox": annotation_2["bbox"],
                "area": annotation_2["area"],
                "iscrowd": annotation_2["iscrowd"]
            }
        }

        # Add the merged annotation to the new list
        new_annotations.append(merged_annotation)
        annotation_id += 1  # Increment the ID counter

    # Prepare the final COCO data
    new_coco_data = {
        "images": images,
        "annotations": new_annotations,
        "categories": categories
    }

    # Save the new COCO data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(new_coco_data, outfile, indent=4)

# Run the function
reformat_coco_data(input_file, output_file)
