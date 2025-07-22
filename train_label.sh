CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_class -c config/coco_class.py --datasets config/coco_class.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased

CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_class/config1 -c config/coco_class1.py --datasets config/coco_class1.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased
CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_class/config2 -c config/coco_class2.py --datasets config/coco_class2.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased

# CUDA_VISIBLE_DEVICES=1 python -u finetune.py --save_results --output_dir ./attribution_finetune/summary_train/best_test/test_pretrain -c config/coco_img.py --eval --datasets config/coco_img.json --resume attribution_img/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased
CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_class/config1/best_test -c config/coco_class1.py --eval --datasets config/coco_class1.json --resume attribution_class/config1/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased
CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_class/config2/best_test -c config/coco_class2.py --eval --datasets config/coco_class2.json --resume attribution_class/config2/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased
CUDA_VISIBLE_DEVICES=0 python -u main.py --save_results --output_dir ./attribution_class/best_test -c config/coco_class.py --eval --datasets config/coco_class.json --resume attribution_class/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased

CUDA_VISIBLE_DEVICES=0 python -u main.py --save_results --output_dir ./attribution_class/config11/best_test -c config/coco_class11.py --eval --datasets config/coco_class11.json --resume attribution_class/config1/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased
CUDA_VISIBLE_DEVICES=1 python -u main.py --save_results --output_dir ./attribution_class/config21/best_test -c config/coco_class21.py --eval --datasets config/coco_class21.json --resume attribution_class/config2/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased

