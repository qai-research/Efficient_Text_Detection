# Data preparation for Efficient Text Detection model training
This README contains:
1. Data preprocessing
2. Data structure for training/testing (LMDB)
3. How to run end-to-end data preparation <br>

Note: The repo is design to run on Unix OS (Ubuntu).
<!-- DATA PREPROCESSING -->
## Data preprocessing
The raw dataset is downloaded and processed as steps in bash (.sh) file. Temporary data download is located in the same dir as .sh file. <br>
Raw dataset then will be extract and convert into the followings format:
### Expected dataset structure for COCO instance/keypoint detection:

```
cocotext/
    {train,test}/
        annotations/
            <img_file_name>.json
        images/
            <img_file_name>.jpg
icdar15/
    ...
synthtext800k/
    synthtext_test
        synthtext_p9
            annotations/
                <img_file_name>.json
            images/
                <img_file_name>.jpg
    synthtext_train
        synthtext_p1/
            annotations/
                    <img_file_name>.json
                images/
                    <img_file_name>.jpg
        ...
        synthtext_p8
    ...
```
<!-- DATA STRUCTURE FOR TRAINING/TESTING-->
## Data structure for training/testing
The datasets for training/testing model is expected to locate in /akaocr/data/. They need to have the following directory structure: <br>
### Expected LMDB dataset structure for training model detection:
```
data/
    train/
        cocotext/
            train/
                ST_train/
                    data.mdb
                    lock.mdb
            test/
                ST_test/
                    ...
        synthtext/
            train/
                ST_synthtext_p1/
                    data.mdb
                    lock.mdb
                ST_synthtext_p*/
                ...
                ST_synthtext_p8/
                    ...
            test
                ST_synthtext_p9
                    data.mdb
                    lock.mdb
        icdar13/
            train/
                ST_train/
                    data.mdb
                    lock.mdb
            test/    
                ST_test/
                    ...
        ...
```
<!-- RUN BASH SCRIPT TO PREPARE LMDB DATASET-->
## Run bash script to prepare LMDB dataset
We create a bash script to auto preprocessing data for training/testing. Script is located in: akaocr/dataprep/prepare.sh<br>
### Run the followings command: <br>
>cd akaocr/dataprep <br>
>sh prepare_detec.sh <br>

After running bash file, successful message "Prepare 4 datasets completed, check at akaocr/data/..." would appear.

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