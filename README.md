
<!-- PROJECT SHIELDS -->
<!--
-->


<!-- PROJECT LOGO -->

# Efficient Text Detection
This repo combine the power of heat map based text detection method with advance backbone from efficient line-up.

## Project Structure

```
├── detec
│   ├── efficient
│   └── craft
└── recog
    └── atten
```

### Built With

* [Python](https://www.python.org/)
* [Torch](https://pytorch.org/)


<!-- GETTING STARTED -->
## Getting Started
### Setup environment
Install pytorch:
>pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

Use requirements.txt (note: use pip/pip3 and python/python3 base on your environment settings)
>cd akaocr/ <br>
>pip3 install -r requirements.txt

If the installation get error or cannot found packages, you may need to upgrade your pip first:
>pip3 install --upgrade pip
### Data detect preparation
Run bash script: <br>
>cd akaocr/dataprep <br>
>sh prepare.sh

### Training
Replace (akaocr) with relative path and match the target train/test set of dataset in akaocr/data. For ex, for ICDAR13:
>--data_detec=(akaocr)/data/icdar13/train <br>
>--data_test=(akaocr)/data/icdar13/test <br>


Train detec: <br>
>cd akaocr/tools

>python3 train_detec.py --data_detec=../data/<dataset_name>/train --data_test_detec=../data/<dataset_name>/test --exp=\<experiment_name> --weight=\<pretrain_model>

Train recog: <br>
>cd akaocr/tools

>python3 train_recog.py --data_recog=../data/<dataset_name>/train --data_test_recog=../data/<dataset_name>/test --exp=\<experiment_name> --weight=\<pretrain_model>
    
### Prerequisites

List of dependencies
* [cuda compatible installation](https://pytorch.org/get-started/locally/)
```
torch==1.5.1
torchvision==0.6.1
```

```
opencv-python==4.4.0.42
tqdm==4.43.0
h5py==2.10.0
imageio==2.8.0
imutils==0.5.3
lmdb==0.98
natsort==7.0.1
nltk==3.5
numpy==1.18.1
pandas==1.0.1
pdf2image==1.13.1
pillow==7.1.2
polygon3==3.0.8
pydot==1.4.1
scipy==1.4.1
scikit-image==0.16.2
tabulate==0.8.6
pygame==2.0.0.dev10
imagecorruptions==1.1.0
imageio==2.8.0
imgaug==0.4.0
```

<!-- LICENSE -->
## License
MIT License

<!-- CONTACT -->
## Contact

Email - [nguyenvietbac1@gmail.com](nguyenvietbac1@gmail.com) <br>
Email - [nnnghia.96@gmail.com](nnnghia.96@gmail.com) <br>
Email - [huukim98@gmail.com](huukim98@gmail.com)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements