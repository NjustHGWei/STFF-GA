import os
import json
import torch
import torch.nn.functional as F
from utils import data_loader
from tqdm import tqdm
from utils.metrics import Evaluator
from module.network import STFFGA

import time
start=time.time()

def test(test_loader, Eva_test, save_path, net , test_path):
    print("Strat validing!")
    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(test_loader)):
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
            target = Y.cpu().numpy()

            # for i in range(output.shape[0]):
            #     probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
            #     final_mask = probs_array * 255
            #     final_mask = final_mask.astype(np.uint8)
            #     final_savepath = save_path + filename[i] + '.png'
            #     im = Image.fromarray(final_mask)
            #     im.save(final_savepath)

            Eva_test.add_batch(target, pred)

    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA=Eva_test.OA()
    Kappa=Eva_test.Kappa()

    metadata = {"Iou:": IoU[1],
                "Pre:": Pre[1],
                "Recall:": Recall[1],
                "F1:": F1[1],
                "OA": OA[1],
                "Kappa": Kappa[1]}


    if not os.path.exists(test_path):
        os.mkdir(test_path)
    with open(test_path + "Result" + '.json', 'w') as fout:
        json.dump(metadata, fout)

    print('[Test] F1: %.4f \n Precision:%.4f \n Recall: %.4f \n OA: %.4f \n Kappa: %.4f \n IoU: %.4f' % (F1[1], Pre[1], Recall[1], OA[1], Kappa[1], IoU[1]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='/root/data1/STFF-GA/checkpoints/LEVIR/best_train.pth', help='model') # best_train
    parser.add_argument('--save_path', type=str, default='', help='test result path')
    parser.add_argument('--test_root', type=str, default='', help='test data path')
    parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--data_name', type=str, default='LEVIR', help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='STFF-GA', help='the test rgb images root')
    parser.add_argument('--test_path', type=str, default='/root/data1/STFF-GA/checkpoints/LEVIR/')
    opt = parser.parse_args()

    opt.save_path = './test_result1/' + opt.data_name + '/'


    if opt.data_name == 'LEVIR':
        opt.test_root = '/root/data1/CD-Data/LEVIR-CD-256/test/'
    elif opt.data_name == 'WHU':
        opt.test_root = '/root/data1/CD-Data/WHU-CD-256/test/'
    elif opt.data_name == 'SYSU':
        opt.test_root = '/root/data1/CD-Data/SYSU-CD256/test/'


    # opt.save_path = opt.save_path + '/'
    test_loader = data_loader.get_test_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=8, shuffle=False, pin_memory=True)
    Eva_test = Evaluator(num_class=2)

    if opt.model_name == 'STFF-GA':
        if torch.cuda.device_count() > 1:
            print("using multi gpu")
            net = STFFGA().cuda()
            model = torch.nn.DataParallel(net, device_ids=[0, 1])  # 多GPU训练
        else:
            model = STFFGA().cuda()
            print('using one gpu')



    if opt.load is not None:

        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)


    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test(test_loader, Eva_test, opt.save_path, model, opt.test_path)

end=time.time()
print('程序测试test的时间为:', end-start)