# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
import torch
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from util.util_jwt import use_op

# parse options
opt = TrainOptions().parse()

# opt.label_nc = len(use_op)

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataset, dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt, dataset)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)
torch.autograd.set_detect_anomaly(True)


for epoch in iter_counter.training_epochs():
  iter_counter.record_epoch_start(epoch)
  for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
    iter_counter.record_one_iteration()

    # Training
    # train generator
    if i % opt.D_steps_per_G == 0:
      trainer.run_generator_one_step(data_i)

    # train discriminator
    if opt.lambda_gan > 0 or opt.lambda_gan_uncond > 0:
      trainer.run_discriminator_one_step(data_i)

    # Visualizations
    if iter_counter.needs_printing():
      losses = trainer.get_latest_losses()

      # for key_now in losses.keys():
      #   plot_fig.plot(key_now, self.loss[key_now])

      visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                      losses, iter_counter.time_per_iter)
      visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

    if iter_counter.needs_displaying():
      visuals = OrderedDict([
        # ('label_list', data_i['label_list']),
        ('synthesized_image', trainer.get_latest_generated()),
        ('input_image', data_i['input_img']),
        ('output_image', data_i['output_img'])])
      visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

    if iter_counter.needs_saving():
      print('saving the latest model (epoch %d, total_steps %d)' %
            (epoch, iter_counter.total_steps_so_far))
      trainer.save('latest', opt)
      iter_counter.record_current_iter()

  trainer.update_learning_rate(epoch)
  iter_counter.record_epoch_end()

  if epoch % opt.save_epoch_freq == 0 or \
      epoch == iter_counter.total_epochs:
    print('saving the model at the end of epoch %d, iters %d' %
          (epoch, iter_counter.total_steps_so_far))
    trainer.save('latest', opt)
    trainer.save(epoch, opt)

print('Training was successfully finished.')


