python -u main.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name wiqa --do_train \
    --data_dir wiqa_data/ \
    --max_seq_length 256 \
    --output_dir ./save_model_tmp \
    --per_gpu_train_batch_size=8 \
    --gradient_accumulation_steps=16 \
    --learning_rate 2e-5 --num_train_epochs 10.0 --weight_decay 0.01 --seed 789 