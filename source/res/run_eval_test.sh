CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./src/config/config_fsdp_t5.yaml ./src/models/test_evaluation.py 	\
	--module 'dst_tod' \
	--test_files './data/interim/GradSearch/SGD/test.json' \
        --batch_size 20 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir './output/GradSearch1309/epoch_4/pytorch_model.bin'\
	--log_input_label_predict './data/interim/GradSearch/SGD/result_test.json'
