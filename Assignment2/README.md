# GNR638: Coding Assignment-2


This repository contains the reproducible code for evaluating representation transfer, fine-tuning strategies, few-shot learning, and corruption robustness using pre-trained CNN backbones (EfficientNet-B0, DenseNet121, ResNet50). 

GNR638Asgn2 is the analysis report based on this assignment.
results folder has all diagrams, which are already included in report.

The random seed is fixed to 42

## How to Run

**1. Prepare the Dataset**
Ensure that the Aerial Images Dataset zip file is saved as train_data.zip in the **same folder** as main.py and extracted so the code can access the image folders. 

**2. Install Requirements**
Be in Assignment2 folder
Install the necessary Python dependencies by running:
pip install -r requirements.txt

**3. Run**
python main.py