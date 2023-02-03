# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data._utils.collate import default_collate


def find_dataset_using_name(dataset_name):
  # Given the option --dataset [datasetname],
  # the file "datasets/datasetname_dataset.py"
  # will be imported.
  dataset_filename = "data." + dataset_name + "_dataset"
  datasetlib = importlib.import_module(dataset_filename)

  # In the file, the class called DatasetNameDataset() will
  # be instantiated. It has to be a subclass of BaseDataset,
  # and it is case-insensitive.
  dataset = None
  target_dataset_name = dataset_name.replace('_', '') + 'dataset'
  for name, cls in datasetlib.__dict__.items():
    if name.lower() == target_dataset_name.lower() \
        and issubclass(cls, BaseDataset):
      dataset = cls

  if dataset is None:
    raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                     "with class name that matches %s in lowercase." %
                     (dataset_filename, target_dataset_name))

  return dataset


def get_option_setter(dataset_name):
  dataset_class = find_dataset_using_name(dataset_name)
  return dataset_class.modify_commandline_options


def collate_fn_buattn(data):
  """注意！default_collate会自动把dict里的tensor进行stack，我们这里只是pack了，没有stack，还需要再过一遍default_collate"""
  # 需要sort吗？怀疑pad_sequence函数会倒序排
  data.sort(key=lambda x: x['glove_words_embed'].size(0), reverse=True)  # pack_padded_sequence 要求要按照序列的长度倒序排列
  words_embed = [x['glove_words_embed'] for x in data]
  paded_words_embed = rnn_utils.pad_sequence(words_embed, batch_first=True, padding_value=0)
  for i, data_dict in enumerate(data):
    data_dict['glove_words_embed'] = paded_words_embed[i]
  data = default_collate(data)
  return data


def collate_fn_bilstm(data):
  """注意！default_collate会自动把dict里的tensor进行stack，我们这里只是pack了，没有stack，还需要再过一遍default_collate"""
  # 需要sort吗？怀疑pad_sequence函数会倒序排
  data.sort(key=lambda x: x['caption_len'], reverse=True)  # pack_padded_sequence 要求要按照序列的长度倒序排列
  words_embed = [x['caption'] for x in data]
  paded_words_embed = rnn_utils.pad_sequence(words_embed, batch_first=True, padding_value=0)
  for i, cur_data in enumerate(data):
    cur_data['caption'] = paded_words_embed[i]

  # 把label_list的长度扩充到一样长
  if 'label_list' in data[0]:
    max_len = max([len(item['label_list']) for item in data])
    zero_label_list = torch.zeros_like(data[0]['label_list'][0])
    for item in data:
      for i in range(max_len - len(item['label_list'])):
        item['label_list'].append(zero_label_list)

  data = default_collate(data)
  return data


def collate_fn_caplen(data):
  """注意！default_collate会自动把dict里的tensor进行stack，我们这里只是pack了，没有stack，还需要再过一遍default_collate"""
  # 需要sort吗？怀疑pad_sequence函数会倒序排
  data.sort(key=lambda x: x['caption_len'], reverse=True)  # pack_padded_sequence 要求要按照序列的长度倒序排列
  data = default_collate(data)
  return data


def create_dataloader(opt):
  dataset = find_dataset_using_name(opt.dataset_mode)
  instance = dataset()
  instance.initialize(opt, mode='train' if opt.isTrain else 'test')
  print("dataset [%s] of size %d was created" %
        (type(instance).__name__, len(instance)))

  # todo: finish collate_fn
  if opt.use_buattn:
    collate_fn = collate_fn_buattn
  elif opt.lang_encoder == 'bilstm':
    collate_fn = collate_fn_bilstm
  else:
    collate_fn = default_collate

  dataloader = torch.utils.data.DataLoader(
    instance,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=int(opt.nThreads),
    drop_last=opt.isTrain,
    collate_fn=collate_fn
  )
  return instance, dataloader
