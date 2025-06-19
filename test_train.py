import argparse
from accelerate import notebook_launcher
from trajlib.runner.base_runner import BaseRunner
import sys
import os
import json
import datetime

# 自定义一个Tee类
class Tee(object):
    def __init__(self, file_name):
        self.file = open(file_name, 'w')  # 打开文件以追加模式
        self.stdout = sys.stdout  # 保存原始标准输出

    def write(self, message):
        self.stdout.write(message)  # 将消息输出到终端
        self.file.write(message)     # 将消息写入文件

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# 配置解析函数
def parse_args():
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--data_form', type=str, default="grid")
    parser.add_argument('--train_mode', type=str, default="pre-train")
    parser.add_argument('--task_name', type=str, default="prediction")
    parser.add_argument('--encoder_name', type=str, default="transformer")
    parser.add_argument('--loss_function', type=str, default="cross_entropy")
    parser.add_argument('--emb_name', type=str, default="normal")
    
    return parser.parse_args()

# 获取命令行参数
args = parse_args()

data_config = {
    "data_name": "chengdu",
    "data_path": "/data/hetianran/didi/chengdu/gps_20161101",
    "data_size": 100000,
    "data_form": args.data_form,
    "grid_step": 100,
    "vocab_size": None,
}

prediction_task_config = {
    "task_name": "prediction",
    "train_mode": args.train_mode, # pre-train, fine-tune, test-only
    "dataset_prop": (0.8, 0.1, 0.1),
    "input_len": 10,
    "output_len": 1,
}

similarity_task_config = {
    "task_name": "similarity",
    "train_mode": args.train_mode, # test-only
    "dataset_prop": (0, 0, 1),
    "variant": "cropped",  # cropped, distorted
    "sub-task": "kNN", # kNN, CDD
}

filling_task_config = {
    "task_name": "filling",
    "train_mode": args.train_mode, # pre-train
    "dataset_prop": (0.9, 0.1, 0),
    "sub-task": "autoregressive",  # mlm, autoregressive
}

embedding_config = {
    "emb_name": args.emb_name,
    "emb_dim": 256,
    "pre-trained": False,
    "embs_path": "",
}

transformer_encoder_config = {
    "encoder_name": "transformer", # transformer, lstm, cnn, mlp
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "dropout": 0.1,
    "num_heads": 8,
    "d_ff": 2048,
}

lstm_encoder_config = {
    "encoder_name": "lstm", # transformer, lstm, cnn, mlp
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "dropout": 0.1,
    "hidden_size": 256,
    "bidirectional": False,
}

cnn_encoder_config = {
    "encoder_name": "cnn", # transformer, lstm, cnn, mlp
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "dropout": 0.1,
    "hidden_size": 256,
    "kernel_size": 5,
}

mlp_encoder_config = {
    "encoder_name": "mlp", # transformer, lstm, cnn, mlp
    "num_layers": 6,
    "d_model": embedding_config["emb_dim"],
    "dropout": 0.1,
    "hidden_size": 256,
}

trainer_config = {
    "log_path": "./resource/model/backbone/backbone.pth",
    "model_path": "./resource/model/backbone/backbone.pth",
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 15,
    "optimizer": "adam",
    "loss_function": args.loss_function,
    "lr_scheduler": "step_lr",
}

config = {
    "data_config": data_config,
    "task_config": None,
    "embedding_config": embedding_config,
    "encoder_config": None,
    "trainer_config": trainer_config,
}

match args.encoder_name:
    case "transformer":
        config["encoder_config"] = transformer_encoder_config
    case "lstm":
        config["encoder_config"] = lstm_encoder_config
    case "cnn":
        config["encoder_config"] = cnn_encoder_config
    case "mlp":
        config["encoder_config"] = mlp_encoder_config
    case _:
        raise ValueError()

match args.task_name:
    case "prediction":
        config["task_config"] = prediction_task_config
    case "similarity":
        config["task_config"] = similarity_task_config
    case "filling":
        config["task_config"] = filling_task_config
    case _:
        raise ValueError()
# config = {
#     "data_config": data_config,
#     "task_config": prediction_task_config,
#     "embedding_config": embedding_config,
#     "encoder_config": cnn_encoder_config,
#     "trainer_config": trainer_config,
# }

output_path = '/home/wangzitong/Trajlib/log'
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_name = f"{timestamp}_{config['task_config']['task_name']}_{config['encoder_config']['encoder_name']}_{config['data_config']['data_form']}"
# 创建日志目录
log_dir = os.path.join(output_path, experiment_name)
os.makedirs(log_dir, exist_ok=True)

# 输出文件路径（例如：日志文件）
output_file = os.path.join(log_dir, "output.log")

# 修改config中的log_path
config['trainer_config']['log_path'] = os.path.join(log_dir, "backbone.pth")

def accelerate_run(config):
    runner = BaseRunner(config)
    runner.run()

# accelerate_run(config)

# 将标准输出重定向到文件
sys.stdout = Tee(output_file)
notebook_launcher(accelerate_run, args=(config,), num_processes=2, use_port="29505")


# 保存config到config.log文件
config_log_file = os.path.join(log_dir, "config.log")
with open(config_log_file, 'w') as f:
    json.dump(config, f, indent=4)

# TODO debug embedding 预训练和 encoder 训练无法连续运行
