
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
### Data preparation
Run bash script: <br>
>sh akaocr/dataprep/prepare.sh

### Training
Replace \<data train/test> with target train/test set of dataset in akaocr/data. For ex:
>--data_detec=akaocr/data/icdar13/train <br>
>--data_test=akaocr/data/icdar13/test <br>


Train detec: <br>
>python akaocr/tools/train_detec.py --data_detec=\<data train> --data_test_detec=\<data test> --exp=\<experiment name> --config=\<config path> --weight=\<pretrain model>

Train recog: <br>
>python akaocr/tools/train_recog.py --data_recog=\<data train> --data_test_recog=\<data test> --exp=\<experiment name> --config=\<config path> --weight=\<pretrain model>
    
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