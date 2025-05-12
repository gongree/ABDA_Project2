# [Advanced Big Data Analysis Project2] MDD Diagnosis with Functional Brain Network

## 🔍 Project Summary
This project explores the use of Graph Structure Learning (GSL) for classifying Major Depressive Disorder (MDD) using resting-state fMRI data.  
The baseline model learns brain network structures from BOLD time-series signals and performs classification using a GNN-based framework.  
The goal is to improve diagnostic performance by leveraging adaptive graph construction techniques or other deep learning methods.

---

## 📂 Data
The preprocessed dataset can be downloaded from the following Google Drive link:  
[Google Drive – fMRI Dataset](<https://drive.google.com/file/d/1aEj5vJleSmbzzoXbolE6pGuXrD33eVNT/view?usp=sharing>)

Once downloaded, please place the `data/` folder in the same directory as the code files.

---

## 💻 Code Structure
```
project2/
│
├── data/
│   └── [your dataset files]
├── datasets.py
├── losses.py
├── main.py
├── model.py
└── utils.py
```

To train and validate the model, simply run:

```bash
python main.py
```
All training settings and hyperparameters can be modified using argparse in main.py.
Run the following to see the full list of options:
```bash
python main.py --help
```

---

If you have any questions, feel free to contact me by email.
