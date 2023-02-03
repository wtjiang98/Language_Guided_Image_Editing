from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


class InfiniteDataloader(object):
  def __init__(self, dataset, opt, collate_fn=default_collate):
    self.opt = opt
    self.dataset = dataset
    self.collate_fn = collate_fn
    loader = DataLoader(dataset, batch_size=self.opt.batch_size,
                        shuffle=False, num_workers=self.opt.num_worker, drop_last=True, collate_fn=self.collate_fn)
    self.len = len(loader)
    self.loader = iter(loader)

  def __iter__(self):
    self.loader = iter(DataLoader(self.dataset, batch_size=self.opt.batch_size,
                                  shuffle=False, num_workers=self.opt.num_worker, drop_last=True,
                                  collate_fn=self.collate_fn))

  def __next__(self):
    try:
      data = next(self.loader)
    except StopIteration:
      self.__iter__()
      data = next(self.loader)
    return data
