import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajlib.data.data import SPECIAL_TOKENS
from trajlib.runner.trainers.base_trainer import BaseTrainer


# mask 左上角 n * n 全为 1，其他为 0
def collate_fn(batch):
    locs, masks = [], []
    for trajs in zip(*batch):
        loc = pad_sequence(trajs, batch_first=True, padding_value=SPECIAL_TOKENS["pad"])
        pad_mask = loc != SPECIAL_TOKENS["pad"]  # (B, L)
        mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # (B, L, L)
        locs.append(loc)
        masks.append(mask.int())
    return locs, masks


class SimilarityTrainer(BaseTrainer):
    def __init__(self, trainer_config, accelerator, model, dataset, geo_data):
        super().__init__(trainer_config, accelerator, model, dataset, geo_data, collate_fn)

    def test(self, epoch):
        self.model.eval()
        all_outputs = []
        with torch.no_grad():
            for locs, masks in tqdm(
                self.test_loader, disable=not self.accelerator.is_local_main_process, desc=f"Final Test"
            ):
                locs = [x.to(self.accelerator.device) for x in locs]
                masks = [m.to(self.accelerator.device) for m in masks]
                outputs = [self.model(loc, mask=mask) for loc, mask in zip(locs, masks)]

                gathered_outputs = self.accelerator.gather_for_metrics(outputs)
                all_outputs.append(gathered_outputs)
        all_outputs = [torch.cat(x, dim=0) for x in zip(*all_outputs)]
        return self._calc_metrics(all_outputs)

    def _calc_metrics(self, outputs: list[torch.Tensor]):
        raise NotImplementedError()


# Most Similar Search
class SimilarityMSSTrainer(SimilarityTrainer):
    def _calc_metrics(self, outputs):
        x_odd, x_even = outputs
        query_ratio = 0.1
        num_query = int(x_odd.size(0) * query_ratio)
        query, db = x_odd[:num_query], x_even

        dists = torch.cdist(query, db)
        sorted_indices = torch.argsort(dists, dim=1)  # (num_query, db_size)
        ground_truth = torch.arange(query.size(0), device=query.device)  # (num_query,)
        match = sorted_indices == ground_truth.unsqueeze(1)  # (num_query, db_size)
        ranks = match.float().argmax(dim=1).float() + 1

        return {"Mean Rank": ranks.mean().item()}


# Cross Distance Deviation
class SimilarityCDDTrainer(SimilarityTrainer):
    def _calc_metrics(self, outputs):
        x_ori, y_ori, x_var, y_var = outputs
        # CDD = |d_var - d_ori| / (d_ori + 1e-6)
        d_ori = torch.linalg.norm(x_ori - y_ori, dim=1)
        d_var = torch.linalg.norm(x_var - y_var, dim=1)
        cdd = torch.abs(d_var - d_ori) / (d_ori + 1e-6)
        return {"Mean CDD": cdd.mean().item()}


# k-Nearest Neighbors
class SimilarityKNNTrainer(SimilarityTrainer):
    def _calc_metrics(self, outputs):
        x_ori, x_var = outputs
        k, query_ratio = 5, 0.1
        num_query = int(x_ori.size(0) * query_ratio)
        query1, db1 = x_ori[:num_query], x_ori[num_query:]
        query2, db2 = x_var[:num_query], x_var[num_query:]

        def knn_search(query, database, k):
            dists = torch.cdist(query, database)  # (num_query, db_size)
            indices = torch.topk(dists, k=k, largest=False).indices  # (num_query, k)
            return indices

        gt_indices = knn_search(query1, db1, k)
        pred_indices = knn_search(query2, db2, k)

        match = pred_indices.unsqueeze(2) == gt_indices.unsqueeze(1)  # (num_query, k, k)

        precision_hits = match.any(dim=2).float().sum()
        precision = precision_hits / (num_query * k)

        recall_hits = match.any(dim=1).float().sum()
        recall = recall_hits / (num_query * k)

        return {f"Precision@{k}": precision.item(), f"Recall@{k}": recall.item()}
