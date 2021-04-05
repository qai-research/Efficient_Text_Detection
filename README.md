
<!-- PROJECT SHIELDS -->
<!--
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
### CI Test Pipeline status
[![Overall Testing Pipeline](https://gitlab.com/cuongvt/ocr-components/badges/develop/pipeline.svg)](https://gitlab.com/cuongvt/ocr-components/-/pipelines)\
Test jobs in detail (see in Pipeline for correctly status, may not real time): 
<br />
[![model test](https://gitlab.com/cuongvt/ocr-components/-/jobs/artifacts/develop/raw/akaocr/test/badge_test_model.svg?job=test-model)](https://gitlab.com/cuongvt/ocr-components/-/jobs)
[![config test](https://gitlab.com/cuongvt/ocr-components/-/jobs/artifacts/develop/raw/akaocr/test/badge_test_config.svg?job=test-config)](https://gitlab.com/cuongvt/ocr-components/-/jobs)
[![dataloader test](https://gitlab.com/cuongvt/ocr-components/-/jobs/artifacts/develop/raw/akaocr/test/badge_test_dataloader.svg?job=test-dataloader)](https://gitlab.com/cuongvt/ocr-components/-/jobs)
[![evaluation test](https://gitlab.com/cuongvt/ocr-components/-/jobs/artifacts/develop/raw/akaocr/test/badge_test_evaluation.svg?job=test-eval)](https://gitlab.com/cuongvt/ocr-components/-/jobs)
[![training loop test](https://gitlab.com/cuongvt/ocr-components/-/jobs/artifacts/develop/raw/akaocr/test/badge_test_train_loop.svg?job=test-train-loop)](https://gitlab.com/cuongvt/ocr-components/-/jobs)

# AkaOCR
OCR pre-train for japanese + custom process to create OCR engine for new project
<br />
<a href="https://git3.fsoft.com.vn/GROUP/DPS/rnd/akaocr"><strong>Explore the docs »</strong></a>
<br />
<a href="https://git3.fsoft.com.vn/GROUP/DPS/rnd/akaocr">View Demo</a>


<!-- ABOUT THE PROJECT -->
## Project Structure

```
├── detec
│   ├── craft
│   └── tesseract
├── recog
│   ├── clova
│   └── tesseract
└── synth
    ├── stdoc
    ├── stimg
    └── stscene
```

### Built With

* [Python](https://www.python.org/)
* [Torch](https://pytorch.org/)
* [Matlab](https://www.mathworks.com/products/matlab.html)



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

List of dependencies
* [cuda compatible installation](https://pytorch.org/)
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

The source code belong to FPT software do not use without authorization

<!-- CONTACT -->
## Contact

Email - [bacnv6@fsoft.com.vn](bacnv6@fsoft.com.vn)

Email - [trangnm5@fsoft.com.vn](trangnm5@fsoft.com.vn)

Project Link: [https://git3.fsoft.com.vn/GROUP/DPS/rnd/akaocr](https://git3.fsoft.com.vn/GROUP/DPS/rnd/akaocr)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=flat-square
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=flat-square
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=flat-square
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=flat-square
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=flat-square
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
[product-screenshot]: images/screenshot.png
