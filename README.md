# MLOps Zoomcamp Final Project - Predict Airline Satisfaction
This project showcases the application of MLOps principles learned in the Data Talks Club MLOps Zoomcamp. The goal is to develop a pipeline that predicts airline customer satisfaction using machine learning models, while leveraging modern MLOps tools and practices.

## Problem Statement

The objective of this project is to build a scalable MLOps pipeline to predict airline customer satisfaction based on several key metrics collected by airlines. The pipeline incorporates the following technologies:

* Environment Creation: Docker
* Infrastructure as Code (IaC): Terraform
* Workflow Orchestration: Prefect
* Cloud Infrastructure: AWS (EC2, S3)
* Experiment Tracking: MLflow
* Model Monitoring: EvidentlyAI

After performing exploratory data analysis (EDA) and comparing various machine learning models in the notebook `flight-satisfaction-prediction-scratch.ipynb`, the CatBoost model was selected for use in the pipeline and further optimized through hyperparameter tuning.

## Data Description

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) and contains responses from an airline passenger satisfaction survey. The key features include:

* Demographics: Gender, Age, Customer Type
* Flight Info: Type of Travel, Class, Flight Distance
* Satisfaction Metrics: Inflight wifi service, Departure/Arrival convenience, Online booking ease, Gate location, Seat comfort, Inflight entertainment, Food & drink, Baggage handling, Legroom, and more
* Delay Metrics: Departure Delay in Minutes, Arrival Delay in Minutes
* Target: Satisfaction (Satisfied, Neutral, Dissatisfied)
* Chosen Model: CatBoost

CatBoost, an open-source gradient boosting algorithm developed by Yandex, was selected for this project due to its superior performance with tabular data, particularly when handling categorical features. The model’s training process is optimized through techniques such as ordered boosting and oblivious trees. After initial testing of various models, CatBoost demonstrated the best performance, which was further improved by hyperparameter tuning using Optuna. All experiments and results were tracked using MLflow.

## Why CatBoost?
* Native handling of categorical features: No need for manual preprocessing.
* Efficiency and accuracy: Proven track record of strong performance in tabular data tasks.
* Robustness: Capable of avoiding overfitting and handling noise in the data effectively.
* MLOps Pipeline Overview

Due to time constraints, certain components like Docker/EvidentlyAI and CI/CD were not fully implemented.

## Pipeline Components
* [Docker](https://www.docker.com/): Ensures consistent environments across development and production by packaging applications in containers.
* [Terraform](https://www.terraform.io/): Automates infrastructure provisioning in the cloud (AWS) via Infrastructure as Code.
* [Prefect](https://www.prefect.io/): Orchestrates the workflow, scheduling, and monitoring tasks in the pipeline.
* [AWS](https://aws.amazon.com/) (EC2, S3): Provides scalable compute and storage resources for the pipeline.
* [MLFlow](https://mlflow.org/): Tracks experiments, models, and performance metrics, ensuring reproducibility and collaboration.
* [EvidentlyAI](https://www.evidentlyai.com/): Planned for model monitoring to ensure performance and fairness in production.

![Pipeline](images/mlops_final.png)

## Detailed Technology Breakdown

### Docker
Docker containers enable consistent, isolated, and lightweight environments, making it easy to develop, deploy, and scale machine learning applications. This technology ensures the application runs the same way in every environment, whether local or in production.

### Terraform
Terraform simplifies the management of cloud infrastructure through code. By using Terraform, the entire cloud environment, including AWS EC2 instances and S3 storage, can be provisioned, managed, and scaled automatically.

### Prefect
Prefect automates workflow orchestration, allowing the scheduling and execution of data pipelines either locally or in the cloud. Prefect's flexibility and ease of integration with existing infrastructure make it a great choice for this project.

### Cloud Infrastructure
AWS EC2 and S3 are used for compute and storage, respectively. EC2 provides virtual servers for running models and workflows, while S3 stores the dataset, model artifacts, and logs.

### MLflow
MLflow tracks experiments, allowing for the versioning and comparison of different models and configurations. This ensures that model development is reproducible and collaborative. MLflow also facilitates the deployment of trained models into production environments.

### EvidentlyAI (Future Work)
EvidentlyAI will be used to monitor models post-deployment, providing insights into model performance and fairness in production.

## Project Setup

To replicate or run this project locally, follow the instructions below. The steps detail the local setup, but some adjustments may be necessary for deployment on AWS.

Create a virtual environment using conda

`conda create --name py35 python=3.10`

Install libraries

`pip install -r ~/mlops-zoomcamp-project/requirements.txt`

### API Keys

The data is available in the `~\data directory`, but you can also source it by creating a Kaggle account and downloading your credentials in a Json format for use with the Kaggle python library.

### Jupyter Notebook

At this point, feel free to play around with the jupyter notebook I provided (`flight-satisfaction-prediction.ipynb`). There's a comparison of several Sklearn classification models and EDA of the data. 

### Setup AWS

Create public and private key for use with EC2:
`ssh-keygen -f <Local directory to store private keys>/mlops_key_pair2 -t rsa -b 4096`

To run the jupyter notebook or another instance on the EC2, you can connect through the public ip address, available on the AWS dashboard:
`ssh -i <Location of private key on local machine> -L localhost:8888:localhost:8888 ubuntu@<EC2 IP Address>`

### Setup Terraform

Set directory, initialize, plan, and apply terraform:

`cd ~/mlops-zoomcamp-project/terraform`

`terraform init`

`terraform plan`

`terraform apply`

### Setup Prefect

To run locally, start prefect server:

`prefect server start`

Set directory, build, apply, and launch prefect agent:

`cd ~/mlops-zoomcamp-project`

`conda activate prefect-env`

`prefect deployment build prefect/main_flow.py:main -n Example_flow`

`prefect deployment apply main-deployment.yaml`

`prefect agent start -q 'default'`

To run on an EC2 instance, follow the additional instructions below:

`prefect config set PREFECT_API_URL=<EC2 Instance Public IP>`

### MLFlow

To run locally:

`mlflow ui --host localhost:5000 --backend-store-uri sqlite:///mlflow.db`

To run on an EC2 instance:

`mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://mlflow:mlflowadmin@...../mlflow_db --default-artifact-root s3://mflow-remote`

After completing the pipeline setup and running Prefect, the final optimized CatBoost model is visible in MLflow. The model achieved an accuracy of 98.9% after optimization using Optuna.

## Conclusion

This project successfully implements a scalable MLOps pipeline for predicting airline satisfaction. While some features like Docker and EvidentlyAI are not yet integrated, the pipeline demonstrates the key aspects of MLOps, from infrastructure setup to model tracking and optimization.
