export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp_t5.yaml src/models/train.py        \
        --module 'dst' \
        --model_name "google/flan-t5-small" \
        --max_target_length 400 \
        --num_train_epochs 5 \
        --output_dir "./output/GradSearch1309/"  \
        --train_files  "./data/interim/GradSearch/SGD/train.json" "./data/interim/GradSearch/FUSEDCHAT/train.json" \
        --val_files   "./data/interim/GradSearch/SGD/val.json" "./data/interim/GradSearch/FUSEDCHAT/val.json" \
        --batch_size  8 \
        --num_beams   4 \
        --weight_decay  0.3 \
        --learning_rate 2e-5 \
        --gradient_accumulation_steps 16 \
        --with_tracking  \
        --report_to wandb \
        --checkpointing_steps epoch \
        --do_eval_per_epoch \
	--max_eval_samples 100
