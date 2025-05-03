# skin-cancer-detection

This is my final year project at university, where I am developing a machine learning model that detects skin cancer from dermatoscopic images.

## Dataset

The dataset used for this project is sourced from the [Harvard Dataverse - HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). It contains a diverse set of dermatoscopic images of skin lesions, each labeled with its corresponding diagnosis.

### How to Download the Dataset

1. Visit the [HAM10000 Dataset on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
 
![Screenshot_2025-05-02_22-59-18](https://github.com/user-attachments/assets/c6276769-a386-42ad-a1a2-ff11fbcd73c2)


2. Scroll down and select the files shown in the image below.

![WhatsApp Image 2025-05-02 at 11 03 58 PM](https://github.com/user-attachments/assets/24c7a4be-dc20-4880-9d73-215f2ae20e69)


3. Extract the downloaded ZIP file to access the images and metadata.

## Directory Structure

Please follow the directory structure below to ensure the project runs correctly:

```text
skin_cancer_image_Detection/       # Repository root
├── CNN/                           # Folder containing model training scripts
├── images/                        # Contains the dataset
│   ├── HAM10000_images_part_1/    # First half of the image dataset
│   └── HAM10000_images_part_2/    # Second half of the image dataset
└── HAM10000_metadata/             # Metadata file for the dataset
```

## Note:

1. Ensure you have at least 30 GB of free disk space before running the project. The dataset and saved models consume a significant amount of storage.
2. After each model run, the model is saved (This allows you to reuse the trained model without retraining from scratch.)


