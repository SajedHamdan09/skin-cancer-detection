# skin-cancer-detection

This is my final year project at university, where I am developing a machine learning model that detects skin cancer from dermatoscopic images.

## Dataset

The dataset used for this project is sourced from the [Harvard Dataverse - HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). It contains a diverse set of dermatoscopic images of skin lesions, each labeled with its corresponding diagnosis.

### How to Download the Dataset

1. Visit the [HAM10000 Dataset on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
 
![ScreenShot](https://raw.githubusercontent.com/SajedHamdan09/skin-cancer-detection/main/setup-images/image-1.png)


2. Scroll down and select the files shown in the image below.

![ScreenShot](https://raw.githubusercontent.com/SajedHamdan09/skin-cancer-detection/main/setup-images/image-2.jpeg)


3. Extract the downloaded ZIP file to access the images and metadata.

## Directory Structure

Please follow the directory structure below to ensure the project runs correctly:

```text
skin-cancer-detection/                               # Repository root
├── CNN/                                             # Folder containing CNN training scripts
├── GDL/                                             # Folder containing GDL training scripts
├── SENN/                                            # Folder containing SENN training scripts
├── TDA/                                             # Folder containing TDA training scripts
├── CNN+TDA/                                         # Folder containing CNN+TDA training scripts
|   ├── new_dataset/                                 # Folder containing new dataset
|   |   ├── image_pre_processing.ipynb               # pre-processing code for new data
|   |   └── ISBI2016_ISIC_Part3B_Training_Data/      # Folder containing image dataset
├── images/                                          # Contains the dataset
│   ├── HAM10000_images_part_1/                      # First half of the image dataset
│   └── HAM10000_images_part_2/                      # Second half of the image dataset
|   image_pre_processing.ipynb                       # pre-processing code
└── HAM10000_metadata/                               # Metadata file for the dataset
```

## Installation

Before running the project, make sure to install the required dependencies using `pip`:

```bash
1. Setup Virtual Environment: python3.10 -m venv myvenv
2. Activate Virtual Enviroment: - source myvenv/bin/activate(linux/macOS)
                                - myvenv\Scripts\activate.bat(Windows)
3. Install Libraries: pip install -r requirements.txt
```

## Note:

1. Ensure you have at least 30 GB of free disk space before running the project. The dataset and saved models consume a significant amount of storage.
2. After each model run, the model is saved (This allows you to reuse the trained model without retraining from scratch.)


