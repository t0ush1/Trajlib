import random
import accelerate
import numpy as np
import torch
import wandb

from trajlib.data.data_factory import create_data
from trajlib.dataset.dataset_factory import create_dataset
from trajlib.model.model_factory import create_model
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
        traj_data, graph_data = create_data(config)
        self.model = create_model(config)
        self.dataset = create_dataset(config, traj_data)
        if graph_data is not None and not config["embedding_config"]["pre-trained"]:
            self.geo_data = graph_data.to_geo_data().to(self.accelerator.device)
        else:
            self.geo_data = None
        self.trainer = create_trainer(config, self.accelerator, self.model, self.dataset, self.geo_data)

        if self.accelerator.is_local_main_process:
            wandb_config = {
                "data_name": config["data_config"]["data_name"],
                "data_form": config["data_config"]["data_form"],
                "task_name": config["task_config"]["task_name"],
                "emb_dim": config["embedding_config"]["emb_dim"],
            }
            wandb.init(project=config["encoder_config"]["encoder_name"], config=wandb_config)

    def run(self):
        for epoch in range(self.config["trainer_config"]["num_epochs"]):
            train_loss = self.trainer.train(epoch)
            val_loss = self.trainer.validate()
            test_loss, test_acc = self.trainer.test()

            self.accelerator.wait_for_everyone()

            if self.accelerator.is_local_main_process:
                wandb.log(
                    {"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss, "test_acc": test_acc},
                    step=epoch,
                )
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}, Test Acc: {test_acc}"
                )

            early_stopping_info = self.trainer.early_stopping(val_loss)
            if early_stopping_info["is_stop"]:
                if self.accelerator.is_local_main_process:
                    print("Early stopping")
                    # TODO 保存模型
                break

            self.trainer.scheduler.step()

        test_loss, test_acc = self.trainer.test()

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            wandb.log({"final_test_loss": test_loss, "final_test_acc": test_acc})
            print(f"Final Test Loss: {test_loss}, Final Test Accuracy: {test_acc}")

        wandb.finish()
