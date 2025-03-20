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
        self.data = create_data(config)
        self.model = create_model(config)
        self.dataset = create_dataset(config, self.data)
        self.accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)
        self.trainer = create_trainer(config, self.accelerator, self.model, self.dataset)

        if self.accelerator.is_local_main_process:
            wandb_config = {
                "data_name": config["data_config"]["data_name"],
                "data_form": config["data_config"]["data_form"],
                "task_name": config["task_config"]["task_name"],
                "d_model": config["model_config"]["d_model"],
                "num_layers": config["model_config"]["num_layers"],
                "num_heads": config["model_config"]["num_heads"],
            }
            wandb.init(project=config["model_config"]["encoder"], config=wandb_config)

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
