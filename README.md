# Neural Architecture Optimization
This is the Pytorch Implementation Code for the Paper [Neural Architecture Optimization](https://arxiv.org/abs/1808.07233).

Authors: [Renqian Luo](http://home.ustc.edu.cn/~lrq)\*, [Fei Tian](https://ustctf.github.io/)\*, [Tao Qin](https://www.microsoft.com/en-us/research/people/taoqin/), [En-Hong Chen](http://staff.ustc.edu.cn/~cheneh/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/). *=equal contribution

## Note
This code is the Pytorch implementation of cnn part of NAO.

This code tries to merge NAO and NAO-WS together, so you can search in both complete way and weight-sharing way, and finally re-train and evaluate the architecture using the same network backbone model.py and follow the same training scheme.

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

## Imagenet

#### To Train Discovered Architectures
You can train the best architecture discovered (show in Fig. 1 in the Appendix of the paper) using:

| Dataset | Script | GPU | Time | Checkpoint| Top1 Error Rate | Top5 Error Rate |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|Imagenet| train_imagenet_4cards.sh | 4 P40 | 6 days | ['Google'](https://drive.google.com/file/d/1zrBJroadBXLv59nbcu78ET_H0IOxVJw-/view?usp=sharing) | 25.7% | 8.2% |

by running:
```
bash train_imagenet_4cards.sh
```

You can train imagenet on N cards, with --batch_size=128*$N, and --lr=0.1*$N

## Acknowledgements
We thank Hieu Pham for the discussion on some details of [`ENAS`](https://github.com/melodyguan/enas) implementation, and Hanxiao Liu for the code base of language modeling task in [`DARTS`](https://github.com/quark0/darts) . We furthermore thank the anonymous reviewers for their constructive comments.
