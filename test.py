# -*- coding: utf-8 -*-
'''
    Created on wed Sept 22 16:46 2018

    Author           : Kailin Xu
    Email            : 10405501@qq.com
    Last edit date   : Nov 9 9:50 2018

South East University Automation College, 211189 Nanjing China
'''

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.RSseti import RSseti
from model.AlexNet import AlexNet

def test(model, root_dir, anno_file, batch_size):
    '''
        Args:
             model          : (nn.Module) untrained darknet
             root_dir       : (string) directory of root
             anno_file      : (string) directory to list file
             batch_size     : (int) batch size
        Returns:
             Output test info
    '''
    data_loader = DataLoader(RSseti(root_dir, anno_file, 28), batch_size=batch_size,
                             shuffle=True)

    cuda = torch.cuda.is_available()

    # MSE loss function
    mseloss = nn.MSELoss()

    # Prepare a txt file to restore loss info
    loss_recorder = open('loss_recorder.txt', 'a+')

    # Perform training process
    for idx, (canvas, target) in enumerate(data_loader):
        if cuda:
            canvas = canvas.cuda()
            target = target.cuda()

        output = model(canvas)
        loss = mseloss(output, target)

        # Calc recall
        nG = len(target)
        output = torch.max(output, 1)[1]
        target = torch.max(target, 1)[1]

        nCorrect = sum(output == target)
        recall = float(nCorrect/nG)

        # Output train info
        print('[Batch %d/%d] [Losses: %f recall: %.5f]' %
                (idx + 1, len(data_loader), loss.item(), recall))
        loss_recorder.write("loss: %f, recall: %f\n" % (loss.item(), recall))

if __name__ == '__main__':
    # Initialize ResNet34
    model = AlexNet()
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    model.load_state_dict(torch.load("AlexNet.pkl"))

    test(model, "D:/KailinXu", "test.txt", 1)
