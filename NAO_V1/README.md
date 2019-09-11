# Neural Architecture Optimization
This is the Pytorch Implementation Code for the Paper [Neural Architecture Optimization](https://arxiv.org/abs/1808.07233).

Authors: [Renqian Luo](http://home.ustc.edu.cn/~lrq)\*, [Fei Tian](https://ustctf.github.io/)\*, [Tao Qin](https://www.microsoft.com/en-us/research/people/taoqin/), [En-Hong Chen](http://staff.ustc.edu.cn/~cheneh/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/). *=equal contribution

## Note
This code is the Pytorch implementation of cnn part of NAO. Also see [`NAO`](https://github.com/renqianluo/NAO) for tensorflow implentation and PTB task

This code tries to merge NAO and NAO-WS together, so you can search in both complete way and weight-sharing way, and finally re-train and evaluate the discovered architecture using the same network backbone model.py and follow the same training scheme.

In order to achieve this, there are few minor differences in some details in the code compared to the original version. 

## License
The codes and models in this repo are released under the GNU GPLv3 license.

## Citation
If you find this work helpful in your research, please use the following BibTex entry to cite our paper.
```
@inproceedings{NAO,
  title={Neural Architecture Optimization},
  author={Renqian Luo and Fei Tian and Tao Qin and En-Hong Chen and Tie-Yan Liu},
  booktitle={Advances in neural information processing systems},
  year={2018}
}

```

_This is not an official Microsoft product._


## Requirment and Dependency
Pytorch >= 1.0.0

## Search by NAO
We do not provide detailed search process here. Please refer to the original repo [`NAO`](https://github.com/renqianluo/NAO/tree/master/NAO).

## To Train Discovered Architecture by NAO
You can train the best architecture we discovered by NAO(show in Fig. 1 in the Appendix of the paper), noted as NAONet-A here.

### CIFAR-10
To train on CIFAR-10, refer to:

| Dataset | Script | GPU | Time | Checkpoint| Top1 Error Rate |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|CIFAR-10| train_NAONet_A_36_cifar10.sh | 1 P40 | 3 days | TBD | 2.48% |
|CIFAR-10| train_NAONet_A_128_cifar10_4cards.sh | 4 P40 | 5 days | [`Google Drive`](https://drive.google.com/file/d/1Xs5ajX-buseRzOZg9gTJ0Vo4u_OEQFff/view?usp=sharing) | 1.93% |

To train NAONet-A-36 (NAONet-A with channel numbers of 36) on single GPU, run:
```
bash train_NAONet_A_36_cifar10.sh
```
To train NAONet-A-128 (NAONet-A with channel numbers of 128) on 4 GPUs, run:
```
bash train_NAONet_A_128_cifar10_4cards.sh
```

### Imagenet
To train on ImageNet, refer to:

| Dataset | Script | GPU | Time | Checkpoint| Top1 Error Rate | Top5 Error Rate |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|Imagenet| train_NAONet_A_imagenet.sh | 1 P40 | 12 days | [`Google Drive`](https://drive.google.com/file/d/1zrBJroadBXLv59nbcu78ET_H0IOxVJw-/view?usp=sharing) | 25.7% | 8.2% |
|Imagenet| train_NAONet_A_imagenet_4cards.sh | 4 P40 | 6 days | [`Google Drive`](https://drive.google.com/file/d/1zrBJroadBXLv59nbcu78ET_H0IOxVJw-/view?usp=sharing) | 25.7% | 8.2% |

To train NAONet-A on single GPU, run:
```
bash train_NAONet_A_imagenet.sh
```
To train NAONet-A on four GPUs, run:
```
bash train_NAONet_A_imagenet_4cards.sh
```

You can train imagenet on N GPUs using the ```train_NAONet_A_imagenet.sh``` script with ```--batch_size=128*$N``` and ```--lr=0.1*$N```

## Search by NAO-WS

To search the CNN architectures for CIFAR-10 with weight sharing, please refer to:
```
./train_search_NAO-WS_cifar10.sh
```

Once the search is done, the final pool of architectures will be in ```exp/search_cifar10/arch_pool``` and the top-5 architectures will be in ```exp/search_cifar10/arch_pool.final```. You can to run them using ```train_NAONet_B_36_cifar10.sh``` and pass in the arch by setting the ```fixed_arc``` argument.

## To Train Discovered Architecture by NAO-WS
You can train the best architecture we discovered by NAO-WS, noted as NAONet-B here.

### CIFAR-10
To train on CIFAR-10, refer to:

| Dataset | Script | GPU | Time | Checkpoint| Top1 Error Rate |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|CIFAR-10| train_NAONet_B_36_cifar10.sh | 1 P40 | 3 days | TBD | 2.93% |

To train NAONet-B-36 (NAONet-B with channel numbers of 36) on single GPU, run:
```
bash train_NAONet_A_36_cifar10.sh
```


## Acknowledgements
We thank Hieu Pham for the discussion on some details of [`ENAS`](https://github.com/melodyguan/enas) implementation, and Hanxiao Liu for the code base of language modeling task in [`DARTS`](https://github.com/quark0/darts) . We furthermore thank the anonymous reviewers for their constructive comments.
