from trajlib.runner.trainers.prediction_trainer import PredictionTrainer
from trajlib.runner.trainers.similarity_trainer import SimilarityCDDTrainer, SimilarityKNNTrainer


def create_trainer(config, accelerator, model, dataset, geo_data):
    task_config = config["task_config"]
    args = config["trainer_config"], accelerator, model, dataset, geo_data

    match task_config:
        case {"task_name": "prediction"}:
            return PredictionTrainer(*args)
        case {"task_name": "similarity", "sub-task": "kNN"}:
            return SimilarityKNNTrainer(*args)
        case {"task_name": "similarity", "sub-task": "CDD"}:
            return SimilarityCDDTrainer(*args)
        case _:
            raise ValueError()
