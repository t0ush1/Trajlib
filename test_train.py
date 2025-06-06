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
    "variant": "cropped",  # cropped, distorted
    "sub-task": "kNN", # kNN, CDD
}

filling_task_config = {
    "task_name": "filling",
    "train_mode": "pre-train",
    "dataset_prop": (0.9, 0.1, 0),
    "sub-task": "autoregressive",  # mlm, autoregressive
}

embedding_config = {
    "emb_name": "normal",
    "emb_dim": 256,
    "pre-trained": False,
    "embs_path": "",
}

encoder_config = {
    "encoder_name": "lstm", # transformer, lstm, cnn, mlp
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "dropout": 0.1,

    ## transformer
    # "num_heads": 8,
    # "d_ff": 2048,

    ## lstm
    "hidden_size": 256,
    "bidirectional": False,

    ## cnn
    # "hidden_size": 256,
    # "kernel_size": 5,

    ## mlp
    # "hidden_size": 256,
}


trainer_config = {
    "model_path": "./resource/model/backbone/backbone.pth",
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "lr_scheduler": "step_lr",
}

config = {
    "data_config": data_config,
    "task_config": prediction_task_config,
    "embedding_config": embedding_config,
    "encoder_config": encoder_config,
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