python3 main_linear.py \
    --dataset cifar100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --split_strategy class \
    --num_tasks 5 \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine
    --lr 1.0 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 7 \
    --name v51 \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project cassle-eval \
    --entity pigpeppa \
    --wandb \
    --save_checkpoint

#    --scheduler step \
#    --lr_decay_steps 60 80 \