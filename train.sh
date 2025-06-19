# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name lstm

# wait()

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode fine-tune --task_name prediction --encoder_name lstm

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name mlp

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode fine-tune --task_name prediction --encoder_name mlp

# wait




# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form gps --train_mode pre-train --task_name prediction --encoder_name transformer --loss_function mse

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form gps --train_mode pre-train --task_name prediction --encoder_name lstm --loss_function mse

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form gps --train_mode pre-train --task_name prediction --encoder_name cnn --loss_function mse

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form gps --train_mode pre-train --task_name prediction --encoder_name mlp --loss_function mse

# wait





# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name prediction --encoder_name transformer --loss_function cross_entropy

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name prediction --encoder_name lstm --loss_function cross_entropy

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name prediction --encoder_name cnn --loss_function cross_entropy

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name prediction --encoder_name mlp --loss_function cross_entropy

# wait





# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name transformer

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name transformer

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name cnn

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name cnn

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name lstm

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name lstm

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name mlp

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name mlp

# wait






CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name transformer

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode test-only --task_name similarity --encoder_name transformer

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name lstm

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode test-only --task_name similarity --encoder_name lstm

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name cnn

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode test-only --task_name similarity --encoder_name cnn

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name mlp

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode test-only --task_name similarity --encoder_name mlp

wait




CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name transformer --emb_name gcn

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode test-only --task_name similarity --encoder_name transformer --emb_name gcn

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode fine-tune --task_name prediction --encoder_name transformer --emb_name gcn

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode pre-train --task_name filling --encoder_name transformer --emb_name gat

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode test-only --task_name similarity --encoder_name transformer --emb_name gat

wait

CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form grid --train_mode fine-tune --task_name prediction --encoder_name transformer --emb_name gat

wait




# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name transformer

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode test-only --task_name similarity --encoder_name transformer

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name transformer

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name lstm

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode test-only --task_name similarity --encoder_name lstm

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name lstm

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name cnn

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode test-only --task_name similarity --encoder_name cnn

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name cnn

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode pre-train --task_name filling --encoder_name mlp

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode test-only --task_name similarity --encoder_name mlp

# wait

# CUDA_VISIBLE_DEVICES=2,3 python test_train.py --data_form roadnet --train_mode fine-tune --task_name prediction --encoder_name mlp

# wait



