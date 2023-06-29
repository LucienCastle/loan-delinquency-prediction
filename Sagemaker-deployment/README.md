## Table of Contents

- [Introduction](#introduction)
- [Creating S3 bucket and Notebook Instance](#creating-s3-bucket-and-notebook-instance)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Deployment and evaluation](#deployment-and-evaluation)
- [Usage](#usage)
- [License](#license)

## Introduction
<p align='justify'>Amazon SageMaker is a machine learning service provided by Amazon Web Services (AWS) that aims to simplify the process of building, 
training, and deploying machine learning models. It provides a fully managed environment for developing and executing machine learning workflows. </p>

 <p align='justify'> SageMaker offers a Jupyter Notebook-like interface called SageMaker Notebook, which allows developers to write and run code in Python. This interface supports
popular machine learning frameworks like TensorFlow, PyTorch, and MXNet, making it easier for developers to leverage their existing knowledge and expertise 
in these frameworks.</p>

<p align='justify'>  With SageMaker, you can create and manage training jobs, tune hyperparameters, track training progress, and deploy trained models to scalable and 
production-ready hosting environments. It also provides capabilities for distributed training, automatic model tuning, and integration with other AWS services.
</p>

## Creating S3 bucket and Notebook Instance
<p align='justify'>To begin, we'll initiate the process by creating an S3 Bucket. This bucket will serve as the storage for our training data and model artifacts once the model 
training is complete. Follow the steps below to create the S3 Bucket:</p>

1. Access the AWS Console and navigate to the S3 service.
2. Click on "Create Bucket" to start the bucket creation process.
3. Select a suitable name for your bucket, ensuring that it is entirely lowercase and adheres to the S3 Bucket Naming rules.
4. Finally, click on the "Create Bucket" button to create the bucket.

<p align='justify'>Once the bucket is created, you can go ahead and go to the SageMaker service and follow these steps to create a Notebook Instance, which will serve as an ML compute instance 
for working with Jupyter Notebooks. SageMaker provides various instance types with different levels of computing and memory capabilities, each at different 
price points. Here's how to create a Notebook Instance:</p>

1. Go to the SageMaker service in the AWS Console.
2. Click on "Notebook Instances."
3. Select "Create Notebook Instance."
4. Choose an appropriate instance type based on your computing and memory requirements.
5. Configure any additional settings as needed.
6. Click on "Create Notebook Instance" to create the instance.

## Data Preprocessing
Create a new notebook with **conda_tensorflow_p36** kernel. Import the required libraries and clean and preprocess the data as done earlier. You can create a simple ETL pipeline. 
Save the train and test sets to `train.csv` and `test.csv` respectively.

## Model Training
Use the `train.py` as the training script which is fed to the SageMaker's Tensorflow Estimator. Follow the code in `loan-delinquency-ann.ipynb` to train the model.

## Deployment and Evaluation
Create a model endpoint and deploy the model to get the inference. Make sure you delete the endpoint afterward to reduce the costs.

## Usage
Open `loan-delinquency-ann.ipynb` and follow the steps. You can change the hyperparameters for the model by passing different parameters to the Tensorflow estimator.

## License
This project is licensed under the [MIT License](LICENSE).
