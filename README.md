# 🧠 Build-Train-Deploy ML Models with Amazon SageMaker

This project demonstrates how to build a production-ready **Multiclass Text Classification** pipeline using **Amazon SageMaker**, from data ingestion and EDA to model deployment via **Lambda** and **API Gateway**.

> 🚀 End-to-end pipeline on AWS: IAM → SageMaker Studio → S3 → PyTorch → Lambda → API Gateway.

---

## 📌 Project Overview

We classify news headlines into categories like **Health**, **Science**, **Business**, and **Entertainment** using a supervised learning model built in **PyTorch** and deployed with SageMaker.

---

## 📁 Dataset

- **Source**: [UCI News Aggregator Dataset](https://archive.ics.uci.edu/dataset/359/news+aggregator)
- **Format**: CSV
- **Uploaded To**: Amazon S3 bucket

---

## 🧰 Tools Used

| Component        | Service                  |
|------------------|---------------------------|
| Notebook IDE     | SageMaker Studio (JupyterLab) |
| Model Training   | PyTorch on SageMaker      |
| Model Storage    | Amazon S3                 |
| Deployment       | SageMaker Endpoint        |
| Inference API    | AWS Lambda + API Gateway  |
| Logs & Metrics   | Amazon CloudWatch         |


## 🧪 Setup & Execution

### 1. ✅ IAM User and Role

- Create an IAM user with **AdministratorAccess**
- Save `.csv` credentials and sign-in URL

### 2. 🧠 SageMaker Studio

- Create a **SageMaker Domain** and launch **Studio**
- Create a new Jupyter notebook

### 3. 📂 Upload Dataset

- Create an S3 bucket and upload the dataset
- Copy the **S3 URI** for use in notebooks

### 4. 📊 Exploratory Data Analysis (EDA)

- Clean and relabel `category` values
- Visualize data distribution


### 5. 🏋️‍♀️ Training
Prepare script.py using PyTorch

Launch a training job using sagemaker.pytorch.estimator

Save output model to S3

### 6. 🚀 Deploy Model
Write inference.py

Deploy SageMaker Endpoint using PyTorchModel

Test predictions inside notebook

### 7. 🌐 API Deployment
Create AWS Lambda function to call SageMaker endpoint

Set up API Gateway with POST route

Test with Postman

