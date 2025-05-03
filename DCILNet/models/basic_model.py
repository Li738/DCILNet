import os

import torch
import numpy as np
from misc.imutils import save_image
from models.networks import *
import matplotlib.pyplot as plt
import  cv2

class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        print(self.device)

        self.checkpoint_dir = args.checkpoint_dir

        self.pred_dir = args.output_folder
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name),
                                    map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        self.G_pred = self.net_G(img_in1, img_in2)
        # self.feature1, self.feature2, self.feature3, self.feature4, self.feature5, self.feature6,self.feature7, self.feature8, self.feature9,self.feature10, self.feature11, self.feature12,self.feature13, self.feature14, self.feature15,self.feature16,self.feature17,self.feature18, self.feature19, self.feature20,self.feature21,self.feature22, self.feature23 = self.net_G(img_in1, img_in2)
        # return self.feature1, self.feature2, self.feature3, self.feature4, self.feature5, self.feature6,self.feature7, self.feature8, self.feature9,self.feature10, self.feature11, self.feature12,self.feature13, self.feature14, self.feature15,self.feature16,self.feature17,self.feature18, self.feature19, self.feature20,self.feature21,self.feature22, self.feature23

        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)

    def _save_predictions2(self):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()  # 形状为 (1, 3, 256, 256)
        names = self.batch['name']  # 假设这里是文件名列表

        # 假设 preds 的形状为 (1, 3, 256, 256)，则只需处理第一项
        for i in range(preds.shape[0]):  # 遍历每个预测结果
            file_name = os.path.join(self.pred_dir, names[i].replace('.jpg', '.png'))

            pred = preds[i]  # 直接使用 preds[i]，形状为 (3, 256, 256)
            pred = pred * 0.5 + 0.5
            pred = pred.detach().cpu().numpy()  # 将预测结果转为 numpy 数组
            pred = np.transpose(pred, (1, 2, 0))

            # 反归一化
            pred = np.clip(pred , 0, 1)
            pred = (pred * 255).astype(np.uint8)  # 转换为 8 位图像
            # 使用 save_image 保存图像
            # 使用 OpenCV 显示图像
            '''cv2.imshow('RGB Image', pred)

            # 等待按键事件并关闭窗口
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''''
            save_image(torch.tensor(pred), file_name)  # 保存为 PNG 文件
