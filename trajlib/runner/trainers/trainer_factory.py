from trajlib.runner.trainers.prediction_trainer import PredictionTrainer
from trajlib.runner.trainers.similarity_trainer import SimilarityTrainer
from trajlib.runner.trainers.filling_trainer import FillingTrainer
from trajlib.runner.trainers.classification_trainer import ClassificationTrainer


def create_trainer(config, accelerator, model, dataset, grid_geo_data, road_geo_data):
    task_config = config["task_config"]
    embedding_config = config["embedding_config"]
    args = config["trainer_config"], accelerator, model, dataset, grid_geo_data, road_geo_data

    match task_config:
        case {"task_name": "prediction", "token": token}:
            return PredictionTrainer(*args, token=token)
        case {"task_name": "similarity", "sub-task": sub_task}:
            return SimilarityTrainer(*args, sub_task=sub_task)
        case {"task_name": "filling", "sub-task": sub_task, "token": token}:
            return FillingTrainer(
                *args,
                sub_task=sub_task,
                token=token,
                grid_vocab_size=embedding_config["grid"]["vocab_size"],
                road_vocab_size=embedding_config["roadnet"]["vocab_size"],
            )
        case {"task_name": "classification"}:
            return ClassificationTrainer(*args)
        case _:
            raise ValueError()
