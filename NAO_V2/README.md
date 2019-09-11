# NAO-V2
This is the Official Implementation Code for the Paper [Understanding and Improving One-shot Neural Architecture Optimization](https://arxiv.org/abs/1808.07233).

Authors: [Renqian Luo](http://home.ustc.edu.cn/~lrq), [Tao Qin](https://www.microsoft.com/en-us/research/people/taoqin/), [En-Hong Chen](http://staff.ustc.edu.cn/~cheneh/)

## License
The codes and models in this repo are released under the GNU GPLv3 license.

## Citation
If you find this work helpful in your research, please use the following BibTex entry to cite our paper.

_This is not an official Microsoft product._


## Requirment and Dependency
Pytorch >= 1.0.0


## Search by NAO-V2

To search the CNN architectures for CIFAR-10 with NAO-V2, please refer to::
```
./train_search_cifar10.sh
```

Once the search is done, the final pool of architectures will be in ```exp/search_cifar10/arch_pool``` and the top-5 architectures will be in ```exp/search_cifar10/arch_pool.final```. You can to run them using ```train_NAONet_V2_36_cifar10.sh``` and pass in the arch by setting the ```fixed_arc``` argument.

## To Train Discovered Architecture by NAO-V2
You can train the best architecture we discovered by NAO-V2, noted as NAONet-V2 here.

### CIFAR-10
To train on CIFAR-10, refer to:

| Dataset | Script | GPU | Time | Checkpoint| Top1 Error Rate  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
|CIFAR-10| train_NAONet_V2_36_cifar10.sh | 1 P40 | 3 days | [`Google Drive`](https://drive.google.com/file/d/1KCCw-sQ1No55aPW4m4d71UlZGCtVBR7E/view?usp=sharing) | 2.60% | 

by running:
```
bash train_NAONet_V2_cifar10.sh
```

### ImageNet
To train on ImageNet, refer to:

| Dataset | Script | GPU | Time | Checkpoint| Top1 Error Rate | Top5 Error Rate |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|Imagenet| train_NAONet_V2_imagenet.sh | 1 P40 | 12 days | TBD | - | - |
|Imagenet| train_NAONet_V2_imagenet_4cards.sh | 4 P40 | 6 days | TBD | - | - |

To train NAONet-V2 on single GPU, run:
```
bash train_NAONet_V2_imagenet.sh
```
To train NAONet-V2 on four GPUs, run:
```
bash train_NAONet_V2_imagenet_4cards.sh
```

You can train imagenet on N GPUs using the ```train_NAONet_V2_imagenet.sh``` script with ```--batch_size=128*$N``` and ```--lr=0.1*$N```
