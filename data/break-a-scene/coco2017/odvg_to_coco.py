import json

# start_img_id = 10
# start_box_id = 1000
def convert_odvg(input_file, output_file):
    start_img_id = 10
    start_box_id = 1000
    # Initialize COCO-style dictionaries
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Category ID tracking to ensure unique categories
    category_mapping = {}
    # all_cats = ['bear', 'cat', 'horse', 'bird', 'dog', 'cow', 'teddy bear', 'zebra', 'sheep', 'elephant']
    # all_cats = ["horse", "bird", "dog", "cat", "cow", "elephant", "bear", "zebra", "teddy bear", "sheep"]
    all_cats = ['dog']
    # annotation_id = 0  # Unique ID for each annotation

    # Read the JSONL file and process each line
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract image information
            # img_id = data["query_file"]["image_id"]
            img_id = start_img_id

            image_info = {
                "id": img_id,
                "file_name": data["filename"],
                "height": data["height"],
                "width": data["width"]
            }
            coco_format["images"].append(image_info)

            # Process each detection instance
            # for instance in data["detection"]["instances"]:
            instance = data["detection"]["instances"][0]
            bbox = instance["bbox"]
            # Convert [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
            bbox_coco = [
                bbox[0], 
                bbox[1], 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1]
            ]

            # Handle categories and ensure unique IDs
            category_name = instance["category"]
            category_id = all_cats.index(category_name) #instance["label"]
            if category_name not in category_mapping:
                category_mapping[category_name] = category_id
                coco_format["categories"].append({
                    "id": category_id,
                    "name": category_name
                })
            # else:
            #     category_id = category_mapping[category_name]

            # Create annotation entry
            query_anno = data["query_file"]
            del query_anno["image_id"]
            del query_anno["label"]
            del query_anno["category"]
            annotation = {
                "id": start_box_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox_coco,
                "area": bbox_coco[2] * bbox_coco[3],
                "iscrowd": 0,
                "query_file": query_anno
            }
            coco_format["annotations"].append(annotation)
            # annotation_id += 1
            start_img_id +=1
            start_box_id +=1
            # break

    # Save the result to a new JSON file in COCO format # ./data/break-a-scene/coco2017/official_data/temp.json
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO annotations saved to {output_file}")

## AttributionGD/data/break-a-scene/coco2017/official_data
convert_odvg('./official_data/val_label.jsonl', './official_data/anno_label.json')