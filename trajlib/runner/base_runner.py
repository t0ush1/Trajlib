import random
import accelerate
import numpy as np
import torch
import wandb

from trajlib.data.data_factory import create_data, load_data
from trajlib.dataset.dataset_factory import create_dataset
from trajlib.model.model_factory import create_model, pretrain_embedding
from trajlib.runner.trainers.trainer_factory import create_trainer


class BaseRunner:
    def __init__(self, config):
        fix_seed = 114514
        random.seed(fix_seed)
        np.random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        accelerate.utils.set_seed(fix_seed)

        self.config = config
        self.accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)

        if self.accelerator.is_local_main_process:
            create_data(config, overwrite=False)
        self.accelerator.wait_for_everyone()

        traj_data, grid_graph_data, road_graph_data = load_data(config)
        self.dataset = create_dataset(config, traj_data)
        self.grid_geo_data = grid_graph_data.to_geo_data()
        self.road_geo_data = road_graph_data.to_geo_data()

        if self.accelerator.is_local_main_process:
            pretrain_embedding(config, self.grid_geo_data, self.road_geo_data, overwrite=False)
        self.accelerator.wait_for_everyone()

        self.model = create_model(config)
        if config["task_config"]["train_mode"] != "pre-train":
            self._load_model()
            for name, param in self.model.named_parameters():
                if not name.startswith("task_head"):
                    param.requires_grad = False
            if self.accelerator.is_local_main_process:
                print("Load model successfully")

        self.trainer = create_trainer(
            config, self.accelerator, self.model, self.dataset, self.grid_geo_data, self.road_geo_data
        )

        if self.accelerator.is_local_main_process:
            # TODO
            wandb_config = {
                "data_name": config["data_config"]["data_name"],
                # "emb_name": config["embedding_config"]["emb_name"],
                "task_name": config["task_config"]["task_name"],
            }
            wandb.init(project=config["encoder_config"]["encoder_name"], config=wandb_config)

    def _save_model(self):
        path = self.config["trainer_config"]["model_path"]
        state_dict = {k: v for k, v in self.model.state_dict().items() if not k.startswith("task_head")}
        torch.save(state_dict, path)

    def _load_model(self):
        path = self.config["trainer_config"]["model_path"]
        state_dict = torch.load(path, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)

    def _log_results(self, results):
        print(", ".join(f"{k}: {v:.4f}" for k, v in results.items()))
        wandb.log(results)

    def run(self):
        epoches = (
            self.config["trainer_config"]["num_epochs"]
            if self.config["task_config"]["train_mode"] != "test-only"
            else 0
        )
        for epoch in range(epoches):
            train_loss = self.trainer.train(epoch)
            val_loss = self.trainer.validate(epoch)
            test_results = self.trainer.test(epoch)
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_local_main_process:
                self._log_results(
                    {
                        "Train Loss": train_loss,
                        "Val Loss": val_loss,
                        **{f"Test {k}": v for k, v in test_results.items()},
                    },
                )

            early_stopping_info = self.trainer.early_stopping(val_loss)
            if early_stopping_info["is_stop"]:
                if self.accelerator.is_local_main_process:
                    print(f"Early stopping in epoch {epoch + 1}")
                break

            self.trainer.scheduler.step()

        test_results = self.trainer.test(-1)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            self._log_results({f"Final {k}": v for k, v in test_results.items()})
            if self.config["task_config"]["train_mode"] == "pre-train":
                self._save_model()
                print("Save model successfully")

        wandb.finish()
