CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py    \
        --module 'res' \
        --test_files './data/interim/GradRes/ORQuAC/test.json' \
        --batch_size 24 \
        --num_beams 4 \
        --with_tracking  \
        --path_to_save_dir './output/GradRes/25.06.wm1000/epoch_38/pytorch_model.bin'\
        --log_input_label_predict './result/GradRes/25_06/orquac.json'