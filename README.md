# DCILNet
Dual-Branch Cross-Resolution Interaction Learning Network for Change Detection at Different Resolutions

The models mm3, mm4, and mm8 correspond to 3x, 4x, and 8x resolution difference network models respectively. When using different models, the corresponding parameters in CD_dataset.py and trainer.py need to be modified.


### Data structure
Change detection data set with pixel-level binary labels；

├─A

├─B

├─label

└─list

A: images of t1 phase(L);

B:images of t2 phase(H);

label: label maps;

list: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.


## Citation

If you use this code for your research, please cite our paper:

@ARTICLE{10816435,
  author={Li, Jinghui and Shao, Feng and Meng, Xiangchao and Yang, Zhiwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Dual-Branch Cross-Resolution Interaction Learning Network for Change Detection at Different Resolutions}, 
  year={2025},
  volume={63},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2024.3523097}
}
