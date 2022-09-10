import torch
import torch.utils.data as DataUtils


class ImgLoader:
    def __init__(self, dataset, batch_size, device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def get_loaders(self, train_valid=True, split_ratio=0.90):
        if train_valid:
            train_size = round(len(self.dataset) * split_ratio)
            valid_size = len(self.dataset) - train_size
            train, valid = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
            train_batch_loader = torch.utils.data.DataLoader2(
                train,
                batch_size=self.batch_size,
                collate_fn=lambda x: tuple(x_.to(self.device) for x_ in torch.utils.data.dataloader.default_collate(x))
            )

            valid_batch_loader = torch.utils.data.DataLoader2(
                valid,
                batch_size=self.batch_size,
                collate_fn=lambda x: tuple(x_.to(self.device) for x_ in torch.utils.data.dataloader.default_collate(x))
            )
            return train_batch_loader, valid_batch_loader

        else:
            batch_loader = torch.utils.data.DataLoader2(
                self.dataset,
                batch_size=self.batch_size,
                collate_fn=lambda x: tuple(x_.to(self.device) for x_ in torch.utils.data.dataloader.default_collate(x))
            )
            return batch_loader
