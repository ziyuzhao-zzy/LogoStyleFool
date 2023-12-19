-- Disclaimer: This GitHub repository is under routine maintenance.
  
# LogoStyleFool

<div align="center">
  <img src="images/Fig1.png" width="800px" />
</div>

This is the source code for our paper "LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer" (AAAI 2024).

## Requirements
+ python == 3.6
+ pytorch == 1.10.0
+ kornia == 0.2.2
+ torchvision == 0.11.3
+ easydict
+ opencv
+ scikit-learn
+ tqdm
+ scipy

## Dataset

Please download the action recognition dataset [UCF-101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads), then process and save them in 'data/'. We use the same preprocessing as [StyleFool](https://github.com/JosephCao0327/StyleFool).

## Pretrained model
The pre-trained model for C3D on UCF-101, as well as models for style transfer, is provided [here](https://1drv.ms/u/s!Aj2hSJitqRWpgVj6TzNI56C7OwhK?e=Ve5kpl).

## Usage

### LogoStyleFool

**Targeted attack**

Run `python main.py --model C3D --dataset UCF101 --video_npy_path ./your/path/BenchPress/v_BenchPress_g20_c06.npy --label 9 --target --target_class 55 --output_path result/`.

**Untargeted attack**

Run `python main.py --model C3D --dataset UCF101 --video_npy_path ./your/path/FrontCrawl/v_FrontCrawl_g09_c03.npy --label 31 --output_path result/`.

**Basic arguments**:
* `--model`: The attacked model.
* `--dataset`: The dataset.
* `--gpu`: ID of the GPU to use.
* `--video_npy_path`: The video path in npy forms.
* `--label`: The label of the video.
* `--target`: Targeted attack or untargeted attack (default).
* `--target_class`: Targeted attack class.
* `--output_path`: The path to save output_adversarial_npy_path.
* `--rl_batch`: The batch size of RL.
* `--steps`: The steps of RL. 
* `--sigma`: The RL reward ratio to control area.
* `--tau`: The RL reward ratio to control distance.
* `--logo_num`: The num of logos.
* `--style_num`: The num of style imgs.
* `--max_iters`: The max iters of LogoS-DCT.
* `--epsilon`: The epsilon of LogoS-DCT.
* `--linf_bound`: The linf bound of perturbation.(0 ~ 1 for LogoStyleFool-$l_2$ and 0 for LogoStyleFool-$l_\infty$)
  

## Acknowledgement
* Part of our implementation is based on [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer), [PatchAttack](https://github.com/Chenglin-Yang/PatchAttack) and [simple-blackbox-attack](https://github.com/cg563/simple-blackbox-attack). We thank for their extraordinary contributions.

## Citation

If you use this code or its parts in your research, please cite the following paper:
```
@inproceedings{cao2024logostylefool,
      title={LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer}, 
      author={Cao, Yuxin and Zhao, Ziyu and Xiao, Xi and Wang, Derui and Xue, Minhui and Lu, Jin},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2024},
      address={Vancouver, Canada},
      month={February}
}
```
