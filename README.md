# C3 and SINet for Lightweight segmentaiton model on Cityscape dataset


## Requirements

- python 3.6
- pytorch >= 0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2.17
- numpy
- tensorflow>=1.13.0
- visdom



## Model

Hyojin Park, Youngjoon Yoo, Geonseok Seo, Dongyoon Han, Sangdoo Yun, Nojun Kwak
" C3: Concentrated-Comprehensive Convolution and its application 
to semantic segmentation "
([paper](https://arxiv.org/abs/1812.04920))

Hyojin Park, Lars Lowe Sjösund, YoungJoon Yoo, Nicolas Monet, Jihwan Bang, Nojun Kwak
" SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial 
Squeeze Modules and Information Blocking Decoder" 
([paper](https://arxiv.org/abs/1911.09099))


|       Model   |  # of Param(M)|   # of Flop(G) | size for Flop |   IoU( val )  |    IoU (test) |   server link   |
| ------------- | ------------- | ------------- | ------------- | ------------- |  ------------- |  ------------- | 
| C3Net[2,3,7,13]|    0.19      |       3.15    |   512*1024    |       66.87   |    64.78      | [link](https://www.cityscapes-dataset.com/anonymous-results/?id=35de55e0c66400ecae916473975cf2e939f8d8af1889e119a9ab8fe70a8147d8) | 
| C3NetV2[2,4,8,16] |    0.18       |       2.66    |   512*1024    |       66.28   |    65.48      | [link](https://www.cityscapes-dataset.com/anonymous-results/?id=b00ac06d2c8e806a236a4a44ecb83e12a0f442419e16650a87082e65736f61ac) | 
| SINet        |    0.12       |       1.22     |   1024*2048    |       68.22   |     66.46    | [link](https://www.cityscapes-dataset.com/anonymous-results/?id=2ce70c4caebe666258a8138c0f296b763bdd160743a75500ed38f7854ff59a68) | 

* C3NetV2 has same encoder structure with C3Net, but uses bilinear upsampling for a decodder structure. 
* SINet is accepted in WACV2020.



## Train
Once you train the model, my code automatically export format for Cityscape Testserver from best training model.
I used P-40 GPU for training. 
C3 and C3_V2 require 2 GPU and SINet needs 1 GPU. 
Train validation txt is for datalodaer function [here](https://github.com/sacmehta/ESPNet/tree/master/train/city)

```shell
python main_multiscale.py -c C3.json

python main_multiscale.py -c C3_V2.json

python main_Auxloss.py -c SINet.json
```

## Citation
If our works is useful to you, please add two papers.
```shell
@article{park2018concentrated,
  title={Concentrated-Comprehensive Convolutions for lightweight semantic segmentation},
  author={Park, Hyojin and Yoo, Youngjoon and Seo, Geonseok and Han, Dongyoon and Yun, Sangdoo and Kwak, Nojun},
  journal={arXiv preprint arXiv:1812.04920},
  year={2018}
}


@article{park2019sinet,
  title={SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder},
  author={Park, Hyojin and Sj{\"o}sund, Lars Lowe and Monet, Nicolas and Yoo, YoungJoon and Kwak, Nojun},
  journal={arXiv preprint arXiv:1911.09099},
  year={2019}
}

```

## Acknowledge
We are grateful to [Clova AI, NAVER](https://github.com/clovaai) with valuable discussions.

I also appreciate my co-authors 
YoungJoon Yoo, Dongyoon Han, Sangdoo Yun and Lars Lowe Sjösund 
from  [Clova AI, NAVER](https://clova.ai/en/research/research-areas.html), 
Nicolas Monet from [NAVER LABS Europe](https://europe.naverlabs.com/)
and Jihwan Bang from [Search Solutions, Inc](https://www.searchsolutions.co.kr/)

I refer ESPNet code for constructing my experiments and also appreciate Sachin Mehta for valuable comments.
Sachin Mehta is [ESPNet](https://github.com/sacmehta/ESPNet) 
and [ESPNetV2](https://github.com/sacmehta/ESPNetv2) author.


## License

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

