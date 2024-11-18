import os

list_exp = [
    "CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_img_and_class/test_negative -c config/coco_img_and_class.py --eval --eval_mode hard_negatives --datasets config/coco_img_and_class.json --resume ./attribution_img_and_class/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased\n",
    "CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_img_and_class/test_negative -c config/coco_img_and_class.py --eval --datasets config/coco_img_and_class.json --resume ./attribution_img_and_class/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased\n",
    # "CUDA_VISIBLE_DEVICES=0 python -u main.py --save_results --output_dir ./attribution_random -c config/coco_random.py --eval --datasets config/coco_random.json --resume ./attribution_random/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased\n"
]

for exp in list_exp:
    with open("run1.sh", "w") as file:
        file.write(exp)
    # bash run_experiments.sh
    os.system('bash run1.sh')