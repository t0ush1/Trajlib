import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from trajlib.data.data import SpecialToken
from trajlib.runner.trainers.base_trainer import BaseTrainer, pad_trajectory


# pad_mask 左上角 n * n 全为 1，其他为 0
def collate_fn(batch):
    trajs, masks = [], []
    for traj_batch in zip(*batch):
        traj = pad_trajectory(traj_batch)
        pad_mask = traj[1] != SpecialToken.PAD  # (B, L)
        mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # (B, L, L)
        trajs.append(traj)
        masks.append(mask.int())
    return trajs, masks


# Most Similar Search
def calc_MSS(x_even, x_odd):
    query_ratio = 0.1
    num_query = int(x_even.size(0) * query_ratio)
    query, db = x_even[:num_query], x_odd

    dists = torch.cdist(query, db)
    sorted_indices = torch.argsort(dists, dim=1)  # (num_query, db_size)
    ground_truth = torch.arange(query.size(0), device=query.device)  # (num_query,)
    match = sorted_indices == ground_truth.unsqueeze(1)  # (num_query, db_size)
    ranks = match.float().argmax(dim=1).float() + 1

    return {"Mean Rank": ranks.mean().item()}


# Cross Distance Deviation
# CDD = |d_var - d_ori| / (d_ori + 1e-6)
def calc_CDD(x_ori, y_ori, x_var, y_var):
    d_ori = torch.linalg.norm(x_ori - y_ori, dim=1)
    d_var = torch.linalg.norm(x_var - y_var, dim=1)
    cdd = torch.abs(d_var - d_ori) / (d_ori + 1e-6)
    return {"Mean CDD": cdd.mean().item()}


# k-Nearest Neighbors
def calc_kNN(x_ori, x_var):
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


class SimilarityTrainer(BaseTrainer):
    def __init__(self, *args, sub_task):
        super().__init__(*args, collate_fn)
        self._calc_metrics = {"MSS": calc_MSS, "CDD": calc_CDD, "kNN": calc_kNN}[sub_task]

    def test(self, epoch):
        self.model.eval()
        all_outputs = []
        with torch.no_grad():
            for trajs, masks in tqdm(
                self.test_loader, disable=not self.accelerator.is_local_main_process, desc=f"Final Test"
            ):
                outputs = [self._call_model(traj, mask=mask) for traj, mask in zip(trajs, masks)]
                gathered_outputs = self.accelerator.gather_for_metrics(outputs)
                all_outputs.append(gathered_outputs)
        all_outputs = [torch.cat(x, dim=0) for x in zip(*all_outputs)]
        return self._calc_metrics(*all_outputs)
