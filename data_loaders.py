import torch.utils.data as DataUtils



class SymbolImgLoader:

    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def get_loader(self):
        img_dataloader = DataUtils.DataLoader2(self.dataset, batch_size=self.batch_size, collate_fn=lambda x: tuple(x_.to(self.device) for x_ in DataUtils.dataloader.default_collate(x)))
        return img_dataloader



# class NaturalImageLoader:
#
#     def __init__(self, dataset, batch_size, device):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.device = device
#
#     def get_loader(self):
#         img_dataloader = DataUtils.DataLoader2(self.dataset, batch_size=self.batch_size, collate_fn=lambda x: tuple(x_.to(self.device) for x_ in DataUtils.dataloader.default_collate(x)))
#         return img_dataloader
