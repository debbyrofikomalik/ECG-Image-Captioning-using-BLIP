# Development of an Automatic Electrocardiogram (ECG) Interpretation System Based on Bootstrapping Language-Image Pre-training (BLIP)

## Project Summary

Cardiovascular disease remains the leading cause of death worldwide, exerting a particularly severe impact in low- and middle-income countries such as Indonesia. To address the shortage of cardiology expertise, especially in remote regions, this project employs the Bootstrapped Language-Image Pre-training (BLIP) vision-language model to automate ECG interpretation by generating diagnostic captions directly from 12-lead ECG images. The BLIP model was fine-tuned using the publicly available PTB-XL dataset across six distinct training scenarios: (1) BLIP-base with 7,000 samples, (2) 14,000 samples, (3) 21,671 samples trained for 6 epochs, (4) 10 epochs, (5) 12 epochs, and (6) BLIP-large with 7,000 samples. Key hyperparameters—including learning rate, weight decay, and scheduler—were kept consistent across all configurations. Model performance was evaluated using standard image captioning metrics, namely BLEU, ROUGE-L, METEOR, CIDEr, and SPICE. The results demonstrate that the BLIP-base model trained on 21,671 samples for 12 epochs achieved the best performance, confirming BLIP’s capability to generate clinically meaningful ECG interpretations and its potential to support clinical decision-making in resource-limited settings.

## Dataset

The PTB-XL ECG dataset is a comprehensive collection comprising 21,837 clinical 12-lead ECG recordings from 18,885 patients, each with a duration of 10 seconds. The raw waveform data was annotated by up to two cardiologists, who assigned one or more diagnostic statements to each recording. A total of 71 distinct ECG statements conforming to the SCP-ECG standard cover diagnostic, morphological, and rhythm-related categories. The dataset is publicly accessible at PhysioNet. This study utilized 12-lead ECG images from PTB-XL available at Kaggle, while textual data was sourced from the previously mentioned PhysioNet repository in the ptbxl_database.csv file.

- **PTB-XL Dataset (CSV):**  
  [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)  
  or  
  [https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)

- **PTB-XL Image Data:**  
  [https://www.kaggle.com/datasets/bjoernjostein/ptb-xl-ecg-image-gmc2024](https://www.kaggle.com/datasets/bjoernjostein/ptb-xl-ecg-image-gmc2024)

**Citation:**  
Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. RRID:SCR_007345.  
[https://doi.org/10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)

## Pre-processing

Prior to training, ECG images were cropped to focus on the signal area, resized to 224×224 pixels to match the Vision Transformer (ViT) input requirements, converted to greyscale, normalized based on overall brightness and contrast statistics, and then saved back in RGB format. Text annotations underwent preprocessing steps including translation to English using the Python library deep-translator, standardization of medical terms and abbreviations, conversion to lowercase, and removal of irrelevant punctuation. The dataset was then divided into subsets of 7,000, 14,000, and 21,671 samples to evaluate the effect of data size, with each subset further split into training, validation, and test sets in an 80:10:10 ratio using a fixed random seed. Image preprocessing procedures are documented in `pre-processing-image-datas.ipynb`, while text preprocessing steps are detailed in `pre-processing-text-datas.ipynb`.

## Training

Model training was conducted under six scenarios: (1) BLIP-base with 7,000 samples, (2) 14,000 samples, (3) 21,671 samples for 6 epochs, (4) 10 epochs, (5) 12 epochs, and (6) BLIP-large with 7,000 samples. Key hyperparameters such as learning rate, weight decay, and scheduler were maintained consistently across scenarios. Training procedures are implemented in `train.ipynb`. For different training scenarios, ECG image datasets should first be uploaded via the “Add Input” feature in Kaggle Notebook or accessed directly by adding the PTB-XL ECG Image (GMC2024) dataset by Bjorn. The same approach applies to the CSV dataset, either by downloading `ptbxl_database.csv` for manual upload or by adding the PTB-XL ECG Dataset by Khyeh via “Add Input.” The file paths can be copied directly by hovering the cursor over the desired file in the Kaggle Data tab and clicking the copy file path icon. These paths should then be defined in the `csv_file` and `image_folder` parameters within `train.ipynb`. The number of training epochs and other parameters, such as train and validation batch sizes, can be adjusted as needed in the configuration dictionary.

## Evaluation

Model performance was evaluated using standard image captioning metrics, including BLEU, ROUGE-L, METEOR, CIDEr, and SPICE, as implemented in `evaluation.ipynb`. After training, a `.pth` model checkpoint is generated. Evaluation can be performed by uploading this `.pth` file using the Kaggle Upload feature, copying the path, and defining it in the `model_path` parameter within `evaluation.ipynb`.

## GUI Development

A web-based graphical user interface (GUI) was developed using Streamlit. Upon accessing the web application, users are greeted with a homepage displaying a header containing the logo and title, along with usage instructions presented in the sidebar. After uploading an ECG image, the system automatically preprocesses the image and displays the diagnostic description in the output section. The GUI implementation is detailed in `app.py`.
