from trajlib.runner.trainers.prediction_trainer import PredictionTrainer
from trajlib.runner.trainers.similarity_trainer import SimilarityMSSTrainer, SimilarityCDDTrainer, SimilarityKNNTrainer
from trajlib.runner.trainers.filling_trainer import FillingTrainer


def create_trainer(config, accelerator, model, dataset, geo_data):
    task_config = config["task_config"]
    args = config["trainer_config"], accelerator, model, dataset, geo_data

    match task_config:
        case {"task_name": "prediction"}:
            return PredictionTrainer(*args)
        case {"task_name": "similarity", "sub-task": "MSS"}:
            return SimilarityMSSTrainer(*args)
        case {"task_name": "similarity", "sub-task": "CDD"}:
            return SimilarityCDDTrainer(*args)
        case {"task_name": "similarity", "sub-task": "kNN"}:
            return SimilarityKNNTrainer(*args)
        case {"task_name": "filling", "sub-task": sub_task}:
            return FillingTrainer(*args, sub_task=sub_task)
        case _:
            raise ValueError()
