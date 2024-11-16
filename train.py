# Thanks to Han et al. for open-sourcing the code on which we completed the STFF-GA network. https://github.com/ChengxiHAN/CGNet-CD

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import utils.visualization as visual
from utils import data_loader
from tqdm import tqdm
import random
from utils.metrics import Evaluator
from utils.loss import dice_bce_loss
from module.network import STFFGA
import time
start = time.time()

torch.cuda.empty_cache()
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, net, criterion, optimizer, num_epoches):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    global best_F1
    epoch_loss = 0
    net.train(True)
    length = 0
    st = time.time()
    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        preds = net(A, B)
        loss = criterion(preds[0], Y) + criterion(preds[1], Y)
        # loss = criterion(preds, Y)

        # ---- loss function ----
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[1])
        # output = F.sigmoid(preds)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = Y.cpu().numpy().astype(int)
        
        Eva_train.add_batch(target, pred)

        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches,\
            train_loss,\
            IoU, Pre, Recall, F1))
    print("Strat validing!")
    new_F1 = F1
    if new_F1 >= best_F1:
        best_F1 = new_F1

        best_train_net = net.state_dict()
        metadata = {"Iou:": IoU,
                    "Pre:": Pre,
                    "Recall:": Recall,
                    "F1:": F1,
                    "Loss:": train_loss}
        # 保存模型和日志 Save model and log
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(save_path + "best_train" + '.json', 'w') as fout:
            json.dump(metadata, fout)
        print('Best train F1 :%.4f; F1 :%.4f' % (IoU, F1))
        torch.save(best_train_net, save_path + "best_train" + '.pth')

    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A, B)[1]
            # preds = net(A, B)
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)
            length += 1
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()

        metadata = {"Iou:": IoU[1],
                    "Pre:": Pre[1],
                    "Recall:": Recall[1],
                    "F1:": F1[1],
                    "Loss:": train_loss}

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(save_path + str(epoch) + '.json', 'w') as fout:
            json.dump(metadata, fout)
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(best_net, save_path + str(epoch) + '.pth')
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()

if __name__ == '__main__':
    seed_everything(42)
    import argparse

    '''
    The WHU dataset, with its relatively small training set, is prone to overfitting. 
    To mitigate this issue, an early stopping strategy can be employed to prevent overfitting.
    
    The SYSU dataset achieves convergence within 50 epochs.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=225, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--data_name', type=str, default='LEVIR', help='the test rgb images root')  # WHU  LEVIR  SYSU
    parser.add_argument('--model_name', type=str, default='STFF-GA', help='the test rgb images root')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    opt = parser.parse_args()


    if opt.data_name == 'WHU':   
        opt.train_root = '/root/data1/CD-Data/WHU-CD-256/train/'
        opt.val_root = '/root/data1/CD-Data/WHU-CD-256/val/'
        print("使用WHU数据集")
    elif opt.data_name == 'LEVIR':
        opt.train_root = '/root/data1/CD-Data/LEVIR-CD-256/train/'
        opt.val_root = '/root/data1/CD-Data/LEVIR-CD-256/val/'
        print("使用LEVIR数据集")
    elif opt.data_name == 'SYSU':
        opt.train_root = '/root/data1/CD-Data/SYSU-CD256/train/'
        opt.val_root = '/root/data1/CD-Data/SYSU-CD256/val/'
        print("使用SYSU数据集")



    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=8, shuffle=False, pin_memory=True)
    Eva_train = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)


    print("是否可用：", torch.cuda.is_available())
    if opt.model_name == 'STFF-GA':
        if torch.cuda.device_count() > 1:
            print("using multi gpu")
            net = STFFGA().cuda()
            model = torch.nn.DataParallel(net, device_ids=[0, 1])  # 多GPU训练
        else:
            model = STFFGA().cuda()
            print('using one gpu')


    criterion = dice_bce_loss().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path + '/' + opt.data_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0
    best_F1 = 0.0

    print("Start train...")

    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        Eva_train.reset()
        Eva_val.reset()
        train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, model, criterion, optimizer, opt.epoch)
        lr_scheduler.step()


end = time.time()
print('程序训练train的时间为:', end-start)