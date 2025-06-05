data_config = {
    "data_name": "chengdu",
    "data_path": "/data/hetianran/didi/chengdu/gps_20161101",
    "data_size": 100000,
    "data_form": "grid",
    "grid_step": 100,
    "vocab_size": None,
}

prediction_task_config = {
    "task_name": "prediction",
    "train_mode": "pre-train", # pre-train, fine-tune, test-only
    "dataset_prop": (0.8, 0.1, 0.1),
    "input_len": 10,
    "output_len": 1,
}

similarity_task_config = {
    "task_name": "similarity",
    "train_mode": "test-only",
    "dataset_prop": (0, 0, 1),
    "sub-task": "kNN", # kNN, CDD
}

embedding_config = {
    "emb_name": "normal",
    "emb_dim": 256,
    "pre-trained": False,
    "embs_path": "",
}

transformer_encoder_config = {
    "encoder_name": "transformer",
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
}

lstm_encoder_config = {
    "encoder_name": "lstm",
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "hidden_size": 256,
    "bidirectional": False,
    "dropout": 0.1,
}

cnn_encoder_config = {
    "encoder_name": "cnn",
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "hidden_size": 256,
    "kernel_size": 5,
    "dropout": 0.1,
}

mlp_encoder_config = {
    "encoder_name": "mlp",
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "hidden_size": 256,
    "dropout": 0.1,
}

trainer_config = {
    "model_path": "./resource/model/backbone/backbone.pth",
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 20,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "lr_scheduler": "step_lr",
}

config = {
    "data_config": data_config,
    "task_config": similarity_task_config,
    "embedding_config": embedding_config,
    "encoder_config": transformer_encoder_config,
    "trainer_config": trainer_config,
}

from accelerate import notebook_launcher
from trajlib.runner.base_runner import BaseRunner

def accelerate_run(config):
    runner = BaseRunner(config)
    runner.run()

# accelerate_run(config)

notebook_launcher(accelerate_run, args=(config,), num_processes=4, use_port="29505")

# TODO debug embedding 预训练和 encoder 训练无法连续运行