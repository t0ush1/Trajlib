import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajlib.runner.trainers.base_trainer import BaseTrainer

# TODO MLM 预训练，自回归预训练


def collate_fn(batch):
    lens, pads = [], []
    for trajs in zip(*batch):
        lengths = torch.tensor([len(t) for t in trajs])
        padded = pad_sequence(trajs, batch_first=True)
        lens.append(lengths)
        pads.append(padded)
    return pads, lens


def make_padding_mask(lengths, max_len=None):
    # 左上方 n*n 为 1，其他为 0
    if max_len is None:
        max_len = lengths.max().item()
    seq_range = torch.arange(max_len, device=lengths.device)
    valid_mask = seq_range.unsqueeze(0) < lengths.unsqueeze(1)
    mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    return mask.int()


class SimilarityTrainer(BaseTrainer):
    def __init__(self, trainer_config, accelerator, model, dataset, geo_data):
        super().__init__(trainer_config, accelerator, model, dataset, geo_data)
        dataset = self.test_loader.dataset
        batch_size = trainer_config["batch_size"]
        self.test_loader = self.accelerator.prepare(
            DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        )  # [x_ori, y_ori, x_var, y_var] or [x_ori, x_var]

    def test(self):
        self.model.eval()
        all_outputs = []
        with torch.no_grad():
            for locs, lens in tqdm(
                self.test_loader, disable=not self.accelerator.is_local_main_process, desc=f"Testing"
            ):
                locs = [x.to(self.accelerator.device) for x in locs]
                lens = [x.to(self.accelerator.device) for x in lens]
                masks = [make_padding_mask(x) for x in lens]
                outputs = [self.model(loc, mask=mask) for loc, mask in zip(locs, masks)]

                gathered_outputs = self.accelerator.gather_for_metrics(outputs)
                all_outputs.append(gathered_outputs)
        all_outputs = [torch.cat(x, dim=0) for x in zip(*all_outputs)]
        return self._calc_metrics(all_outputs)

    def _calc_metrics(self, outputs: list[torch.Tensor]):
        raise NotImplementedError()


class SimilarityCDDTrainer(SimilarityTrainer):
    def _calc_metrics(self, outputs):
        x_ori, y_ori, x_var, y_var = outputs
        # CDD = |d_var - d_ori| / (d_ori + 1e-6)
        d_ori = torch.linalg.norm(x_ori - y_ori, dim=1)
        d_var = torch.linalg.norm(x_var - y_var, dim=1)
        cdd = torch.abs(d_var - d_ori) / (d_ori + 1e-6)
        return {"Mean CDD": cdd.mean().item()}


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

        return {f"Precision@{k}": precision, f"Recall@{k}": recall}
