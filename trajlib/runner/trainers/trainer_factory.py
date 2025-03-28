from trajlib.runner.trainers.base_trainer import BaseTrainer


def create_trainer(config, accelerator, model, dataset, geo_data):
    return BaseTrainer(config["trainer_config"], accelerator, model, dataset, geo_data)
