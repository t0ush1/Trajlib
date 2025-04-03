data_config = {
    "data_name": "chengdu",
    "data_path": "/data/hetianran/didi/chengdu/gps_20161101",
    "data_size": 1000,
    "data_form": "roadnet",
    # "grid_step": 100,
}

task_config = {
    "task_name": "prediction",
    "input_len": 10,
    "output_len": 1,
}

dataset_config = {
    "val_prop": 0.1,
    "test_prop": 0.1,
}

model_config = {
    "embedding": "node2vec",
    "embs_path": "./resource/model/embedding/node2vec.pkl",
    "vocab_size": None,
    "encoder": "transformer",
    "num_layers": 6,
    "d_model": 512,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
}

trainer_config = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "lr_scheduler": "step_lr",
}

config = {
    "data_config": data_config,
    "task_config": task_config,
    "dataset_config": dataset_config,
    "model_config": model_config,
    "trainer_config": trainer_config,
}

from accelerate import notebook_launcher
from trajlib.runner.base_runner import BaseRunner


def accelerate_run(config):
    runner = BaseRunner(config)
    runner.run()

notebook_launcher(accelerate_run, args=(config,), num_processes=4)