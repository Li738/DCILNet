from argparse import ArgumentParser

import utils
import torch
from models.basic_model import CDEvaluator
import scipy.io
import os
import torch
torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(0)
torch.cuda.is_available()
torch.cuda.current_device()
print(torch.cuda.current_device())
"""
quick start

sample files in ./samples

save prediction files in the ./samples/predict

"""


def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='D:/LJH/Bitmm2/checkpoints/MMLEVIRX8_MM/', type=str)
    parser.add_argument('--output_folder', default='samples_LEVIR/LEVIRX8_tezheng', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--net_G', default='mmNet8', type=str,
                        help='ChangeFormerV6 | CD_SiamUnet_diff | SiamUnet_conc | Unet | DTCDSCN | base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids) > 0
                          else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')

    data_loader = utils.get_loader(args.data_name, img_size=args.img_size,

                                   batch_size=args.batch_size,
                                   split=args.split, is_train=False)

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()
    with torch.no_grad():
     for i, batch in enumerate(data_loader):
        name = batch['name']
        print('process: %s' % name)
        feature1, feature2, feature3, feature4, feature5, feature6,feature7, feature8, feature9,feature10, feature11, feature12,feature13, feature14, feature15,feature16,feature17,feature18,feature19,feature20,feature21,feature22,feature23 = model._forward_pass(batch)
        feature_1 = feature1.cpu()
        feature01 = feature_1.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature1.mat',{'features': feature01})
        feature_2 = feature2.cpu()
        feature02 = feature_2.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature2.mat', {'features': feature02})
        feature_3 = feature3.cpu()
        feature03 = feature_3.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature3.mat', {'features': feature03})
        feature_4 = feature4.cpu()
        feature04 = feature_4.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature4.mat', {'features': feature04})
        feature_5 = feature5.cpu()
        feature05 = feature_5.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature5.mat', {'features': feature05})
        feature_6 = feature6.cpu()
        feature06 = feature_6.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature6.mat', {'features': feature06})
        feature_7 = feature7.cpu()
        feature07 = feature_7.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature7.mat', {'features': feature07})
        feature_8 = feature8.cpu()
        feature08 = feature_8.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature8.mat', {'features': feature08})
        feature_9 = feature9.cpu()
        feature09 = feature_9.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature9.mat', {'features': feature09})
        feature_10 = feature10.cpu()
        feature010 = feature_10.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature10.mat', {'features': feature010})
        feature_11 = feature11.cpu()
        feature011 = feature_11.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature11.mat', {'features': feature011})
        feature_12 = feature12.cpu()
        feature012 = feature_12.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature12.mat', {'features': feature012})
        feature_13 = feature13.cpu()
        feature013 = feature_13.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature13.mat', {'features': feature013})
        feature_14 = feature14.cpu()
        feature014 = feature_14.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature14.mat', {'features': feature014})
        feature_15 = feature15.cpu()
        feature015 = feature_15.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature15.mat', {'features': feature015})
        feature_16 = feature16.cpu()
        feature016 = feature_16.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature16.mat', {'features': feature016})
        feature_17 = feature17.cpu()
        feature017 = feature_17.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature17.mat', {'features': feature017})
        feature_18 = feature18.cpu()
        feature018 = feature_18.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature18.mat', {'features': feature018})
        feature_19 = feature19.cpu()
        feature019 = feature_19.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature19.mat', {'features': feature019})
        feature_20 = feature20.cpu()
        feature020 = feature_20.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature20.mat', {'features': feature020})
        feature_21 = feature21.cpu()
        feature021 = feature_21.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature21.mat', {'features': feature021})
        feature_22 = feature22.cpu()
        feature022 = feature_22.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature22.mat', {'features': feature022})
        feature_23 = feature23.cpu()
        feature023 = feature_23.numpy()
        scipy.io.savemat('D:/LJH/SHUJUJI/LEVIR8/relitu2/feature23.mat', {'features': feature023})


        model._save_predictions()







