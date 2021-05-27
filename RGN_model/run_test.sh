python main.py \
    --model_type roberta \
    --model_name_or_path ./saved_model/ \
    --task_name wiqa --do_test \
    --data_dir ./wiqa_data/ \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=8 \
    --weight_decay 0.01 \
    --output_dir ./saved_model/ \
    --seed 789 