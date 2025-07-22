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
    # all_cats = ['dog']
    # all_cats = ["teddy bear", "zebra", "horse", "cow", "elephant", "dog", "sheep", "cat", "bear", "bird"]
    
    # val cats ------------------------------------------------------------------------------------------------
    # all_cats = ['bench', 'scarf', 'choker', 'cigarette', 'boat', 'polo_shirt', 'short_pants', 'zebra', 'waffle_iron', 'bottle', 'plate', 'sock', 'soccer_ball', 'banana', 'baseball_cap', 'flower_arrangement', 'cake', 'napkin', 'fork', 'spoon', 'train_(railroad_vehicle)', 'poster', 'hat', 'beanie', 'cornet', 'tights_(clothing)', 'ski', 'glove', 'toy', 'toothbrush', 'surfboard', 'teddy_bear', 'jacket', 'umbrella', 'billboard', 'car_(automobile)', 'trousers', 'belt', 'streetlight', 'jean', 'birthday_cake', 'Christmas_tree', 'beanbag', 'pillow', 'lamppost', 'curtain', 'clock_tower', 'signboard', 'snowboard', 'shovel', 'laptop_computer', 'refrigerator', 'tray', 'bucket', 'mask', 'helmet', 'sheep', 'handbag', 'boot', 'water_bottle', 'necklace', 'jar', 'bowl', 'bread', 'paper_plate', 'motorcycle', 'baseball_bat', 'fire_engine', 'hose', 'crossbar', 'table', 'computer_keyboard', 'pastry', 'toilet', 'cellular_telephone', 'fireplace', 'cookie', 'cow', 'blanket', 'dog', 'log', 'knee_pad', 'pickup_truck', 'condiment', 'cat', 'tennis_racket', 'visor', 'tennis_ball', 'stove', 'saucepan', 'pan_(for_cooking)', 'giraffe', 'manger', 'sweater', 'wineglass', 'tote_bag', 'swimsuit', 'shoe', 'jersey', 'parking_meter', 'coat', 'horse', 'saddle_(on_an_animal)', 'bear', 'statue_(sculpture)', 'clock', 'bullet_train', 'slide', 'ski_parka', 'truck', 'trash_can', 'dress', 'skateboard', 'strainer', 'flowerpot', 'pizza', 'person', 'wine_bottle', 'bus_(vehicle)', 'wheel', 'vent', 'bun', 'tank_top_(clothing)', 'shirt', 'painting', 'vase', 'sink', 'dishwasher', 'faucet', 'wall_socket', 'bath_mat', 'banner', 'sofa', 'recliner', 'coffee_table', 'television_set', 'sweatshirt', 'soup', 'broccoli', 'bandanna', 'bed', 'bow_(decorative_ribbons)', 'fireplug', 'saucer', 'tongs', 'bicycle', 'street_sign', 'sail', 'mast', 'vest', 'stool', 'mixer_(kitchen_tool)', 'necktie', 'monitor_(computer_equipment) computer_monitor', 'polar_bear', 'carrot', 'dispenser', 'elephant', 'book', 'wristlet', 'ski_boot', 'kite', 'ski_pole', 'bird', 'mirror', 'suitcase', 'box', 'spatula', 'apple', 'pear', 'chair', 'desk', 'bulldozer', 'rat', 'brake_light', 'medicine', 'license_plate', 'cowboy_hat', 'flag', 'tablecloth']
    
    # test cats ------------------------------------------------------------------------------------------------
    all_cats = ['zebra', 'elephant', 'boat', 'flag', 'mast', 'cow', 'choker', 'flower_arrangement', 'vase', 'place_mat', 'sofa', 'curtain', 'chandelier', 'skirt', 'chair', 'backpack', 'shoe', 'avocado', 'knife', 'cucumber', 'bathtub', 'plastic_bag', 'sink', 'box', 'surfboard', 'balloon', 'carton', 'glass_(drink_container)', 'toilet', 'cistern', 'cover', 'trousers', 'truck', 'baby_buggy', 'polo_shirt', 'short_pants', 'tennis_racket', 'tank_top_(clothing)', 'shopping_cart', 'banana', 'baseball_base', 'snowboard', 'banner', 'cellular_telephone', 'poster', 'tray', 'mug', 'table', 'pancake', 'blackberry', 'pipe', 'faucet', 'necktie', 'monitor_(computer_equipment) computer_monitor', 'mouse_(computer_equipment)', 'computer_keyboard', 'sweatshirt', 'train_(railroad_vehicle)', 'ball', 'log', 'airplane', 'saucer', 'boot', 'pizza', 'radiator', 'microwave_oven', 'orange_(fruit)', 'stop_sign', 'car_(automobile)', 'signboard', 'suitcase', 'shopping_bag', 'giraffe', 'bird', 'chopping_board', 'carrot', 'celery', 'bowl', 'ski', 'ski_pole', 'laptop_computer', 'jersey', 'baseball_glove', 'belt', 'bed', 'blanket', 'cupboard', 'oven', 'refrigerator', 'vent', 'armchair', 'loveseat', 'fireplace', 'street_sign', 'cat', 'handle', 'person', 'umbrella', 'handbag', 'sandwich', 'tomato', 'coat', 'lamppost', 'tablecloth', 'lanyard', 'teddy_bear', 'headboard', 'slipper_(footwear)', 'lamp', 'lampshade', 'horse', 'bear', 'ski_boot', 'bus_(vehicle)', 'windshield_wiper', 'wheel', 'tag', 'taillight', 'propeller', 'plow_(farm_equipment)', 'broccoli', 'jean', 'apple', 'bracelet', 'mound_(baseball)', 'dog', 'frisbee', 'duffel_bag', 'tennis_ball', 'antenna', 'skateboard', 'blinder_(for_horses)', 'book', 'strap', 'camera_lens', 'hog', 'apron', 'hat', 'pacifier', 'Lego', 'onion', 'bench', 'ski_parka', 'mirror', 'jacket', 'printer', 'wall_clock', 'telephone_pole', 'beer_bottle', 'globe', 'ladder', 'desk', 'pillow', 'basket', 'lightbulb', 'coat_hanger', 'drawer', 'ambulance', 'cone', 'pad', 'crossbar', 'goggles', 'glove', 'vest', 'license_plate', 'motorcycle', 'blinker', 'kite', 'headband', 'manger', 'pole', 'pickle', 'cup', 'traffic_light', 'dress', 'chopstick', 'bottle_cap', 'shirt', 'hinge', 'bun', 'stapler_(stapling_machine)', 'doughnut', 'statue_(sculpture)', 'fan', 'baseball_cap', 'bolt', 'fireplug', 'flagpole', 'bicycle', 'wet_suit', 'shower_curtain', 'bath_mat', 'bath_towel', 'gazelle']
    # annotation_id = 0  # Unique ID for each annotation
    print(len(all_cats))

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
convert_odvg('./official_data/test_filename.jsonl', './official_data/anno_test_filename.json')