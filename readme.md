# Interpretable Deep Learning for Multi-Stage Acute Lymphoblastic Leukemia Using DenseNet

## Description
This project implements a DenseNet model to classify images of blood smears into four stages of Acute Lymphoblastic Leukemia (ALL): benign, early, pre, and pro. It emphasizes interpretability by using Grad-CAM to visualize the features the model focuses on for each stage, providing insights into the decision-making process. The pipeline includes data preprocessing, model training, evaluation, and visualization of results.

## Project Structure

      project/
      │
      ├── data/               # raw and processed images
      ├── models/             # saved DenseNet checkpoints
      ├── notebooks/          # preprocessing, training, evaluation
      ├── results/            # plots, Grad-CAM visualizations, confusion matrices
      └── README.md           # this file
      └── requirements.txt    # intstallation requirements

## Dataset

| Path             | Subclass | Description                     |
|--------------|----------|-------------------------------------|
| `/benign`    | Benign   | Non-cancerous, healthy cells        |
| `/early`     | Early    | Early stages of leukemia            |
| `/pre`       | Pre      | Pre-stage abnormal cells            |
| `/pro`       | Pro      | Advanced leukemia cells             |

> These images are used for training a DenseNet model to classify multi-stage ALL. This project emphasizes **interpretability** via Grad-CAM visualizations. 

## Usage

To reproduce the results:

1. **Prepare the dataset**  
   Run `01_data_prep.ipynb` to load, clean, and preprocess images for training and evaluation.

2. **Train the model**  
   Run `02_training_densenet.ipynb` to train DenseNet on the multi-stage ALL dataset.

3. **Evaluate performance**  
   Run `03_evaluation.ipynb` to generate metrics (accuracy, precision, recall, F1-score), confusion matrices, and Grad-CAM visualizations for interpretability.

## Results

- Confusion matrices per stage  
- Class-wise performance metrics  
- Grad-CAM visualizations highlighting leukocyte features per stage  

> All results are stored in the `results/` folder.

## Citations

**Data Citation**  
Mehrad Aria, Mustafa Ghaderzadeh, Davood Bashash, Hassan Abolghasemi, Farkhondeh Asadi, and Azamossadat Hosseini, “Acute Lymphoblastic Leukemia (ALL) image dataset.” Kaggle, 2021. DOI: [10.34740/KAGGLE/DSV/2175623](https://www.kaggle.com/datasets/mohammadmahdi/acute-lymphoblastic-leukemia-all-image-dataset)

**Publication Citation**  
Ghaderzadeh, M., Aria, M., Hosseini, A., Asadi, F., Bashash, D., Abolghasemi, H. A fast and efficient CNN model for B-ALL diagnosis and its subtypes classification using peripheral blood smear images. *Int J Intell Syst.*, 2022; 37: 5113–5133. doi:[10.1002/int.22753](https://doi.org/10.1002/int.22753)

**Dataset Credit**  
Obuli Sai Naren. (2022). Multi Cancer Dataset [Data set]. Kaggle. [https://doi.org/10.34740/KAGGLE/DSV/3415848](https://doi.org/10.34740/KAGGLE/DSV/3415848)

## Notes

- This project is for research purposes only and **not intended for clinical diagnosis**.  
- Grad-CAM visualizations provide interpretability but should **not replace expert evaluation**.  
- Make sure all dependencies (PyTorch, torchvision, matplotlib, etc.) are installed.
