export NCCL_DEBUG=INFO
WANDB_API_KEY=0b39e2667775768150d99b18418fb63ca15b13bc \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /kaggle/working/aip/source/res/src/config/config_fsdp_t5.yaml /kaggle/working/aip/source/res/src/train.py        \
        --module 'res' \
        --model_name "google/flan-t5-base" \
        --max_target_length 400 \
        --num_train_epochs 45 \
        --output_dir "./output/GradRes/30.12"  \
        --train_files  "/kaggle/input/fusedchaat-res/FUSHEDCHAT - Copy/train/fusedchat_train.json"\
        --val_files  "/kaggle/input/fusedchaat-res/FUSHEDCHAT - Copy/validation/fusedchat_val.json"\
        --resume_from_checkpoint "/kaggle/input/ckpt-res/epoch_22" \
        --batch_size  6 \
        --num_beams  4 \
        --weight_decay  0.3 \
        --learning_rate 3e-5 \
        --num_warmup_steps 1500 \
        --gradient_accumulation_steps 16 \
        --with_tracking  \
        --report_to wandb \
        --checkpointing_steps epoch \
        --do_eval_per_epoch