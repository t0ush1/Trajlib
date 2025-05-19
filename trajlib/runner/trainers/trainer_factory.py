from trajlib.runner.trainers.prediction_trainer import PredictionTrainer
from trajlib.runner.trainers.similarity_trainer import SimilarityCDDTrainer, SimilarityKNNTrainer


def create_trainer(config, accelerator, model, dataset, geo_data):
    task_config = config["task_config"]

    if task_config["task_name"] == "prediction":
        constructor = PredictionTrainer

    elif task_config["task_name"] == "similarity":
        if task_config["sub-task"] == "kNN":
            constructor = SimilarityKNNTrainer
        elif task_config["sub-task"] == "CDD":
            constructor = SimilarityCDDTrainer

    return constructor(config["trainer_config"], accelerator, model, dataset, geo_data)
