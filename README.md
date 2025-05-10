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



1. **For the LEVIR-CD (8×) dataset**:  
   The inputs are T1 (32×32) and T2 (256×256), with low-resolution labels at 32×32 and high-resolution labels at 256×256. The **mm8** network model is employed.

2. **For the Google Dataset (4×)**:  
   The inputs are T1 (64×64) and T2 (256×256), with low-resolution labels at 64×64 and high-resolution labels at 256×256. The **mm4** network model is used.

3. **For the DE-CD (3.3×) dataset**:  
   The inputs are T1 (256×256) and T2 (256×256). Due to their 3.3× resolution difference, to align with the network model and enable multiple consecutive downsampling operations, T1 is internally downsampled to an appropriate 64×64 within the network. Correspondingly, the low-resolution labels are downsampled to 64×64, while the high-resolution labels remain at 256×256. The **mm3**network model is adopted.


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
