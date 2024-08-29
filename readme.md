# Fraud Detection Using Graph Neural Networks (GNNs)

## Project Overview

Fraud detection is a critical process in industries such as banking, finance, and e-commerce. This project leverages Graph Neural Networks (GNNs) to detect fraudulent transactions within a dataset of credit card transactions. Traditional fraud detection methods often struggle with the complexity and evolving nature of fraudulent activities. GNNs, however, offer a novel approach by considering the relationships between transactions in a graph structure, making it possible to identify complex fraud patterns that would otherwise go unnoticed.

## Aim of the Project

The primary goal of this project is to develop a machine learning model capable of detecting fraudulent transactions using Graph Neural Networks (GNNs). The project also explores the use of GNNs for their ability to capture complex and evolving fraud patterns, improve detection accuracy, and provide model interpretability, which is essential for transparency and regulatory compliance.

## Dataset

The dataset used in this project is the **IBM Credit Card Transaction Dataset**, which is publicly available. The dataset includes:
- **24 million unique transactions**
- **6,000 unique merchants**
- **100,000 unique cards**
- **30,000 fraudulent samples (0.1% of total transactions)**

**Dataset Source**: [IBM Credit Card Transaction Dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)

### Key Features in the Dataset:
- **User ID** and **Card ID** (combined into a unique identifier `card_id`)
- **Transaction Amount**
- **Use of Chip** (Yes/No)
- **Merchant Details** (Name, City, State, Zip Code)
- **Transaction Timestamp** (Year, Month, Day, Hour, Minute)
- **Error Codes** (e.g., Bad PIN, Insufficient Balance)
- **Fraud Label** (indicating whether the transaction was fraudulent)

## Libraries and Tools

The project makes use of several libraries, including:

- **Python Libraries**: 
  - `pandas`, `numpy` for data manipulation and preprocessing
  - `matplotlib` for data visualization
  - `sklearn` for data processing and model evaluation
  - `networkx` for graph construction
  - `torch` for building and training the GNN model
  - `inductiveGRL` (for method 2 based on the Inductive Graph Representation Learning framework)
  - `stellargraph` for graph-based machine learning tasks

## Project Steps

### 1. **Data Preprocessing**
   - Loaded the dataset and sampled 100,000 transactions due to hardware limitations.
   - Cleaned the data by handling missing values and encoding categorical variables.
   - Created a unique identifier (`card_id`) by combining User and Card IDs.
   - Preprocessed transaction amounts, timestamps, and other relevant features.

### 2. **Graph Construction**
   - Represented transactions as a graph with `card_id` and `merchant_name` as nodes, and transactions as edges.
   - Created a multigraph to handle multiple transactions between the same user and merchant.
   - Added transaction details as edge attributes (e.g., amount, time, errors).

### 3. **Model Development**
   - **Method 1**: Implemented a simple Graph Neural Network (GNN) using PyTorch, focusing on edge classification to detect fraudulent transactions.
   - **Method 2**: Leveraged the Inductive Graph Representation Learning framework (`inductiveGRL`) to create a more advanced GNN model using the GraphSAGE architecture.
   - Trained the models using a binary classification approach, where the target is whether a transaction is fraudulent.

### 4. **Model Training and Evaluation**
   - Split the dataset into training (70%) and inductive (30%) sets.
   - Trained the GNN models on the training set, and evaluated their performance on the inductive set.
   - Assessed model performance using loss metrics and visualized the results.

## Results and Insights

The GNN models effectively learned to detect fraudulent transactions by leveraging the complex relationships within the transactional graph. The project demonstrates the potential of GNNs in improving fraud detection systems, particularly in handling sophisticated and evolving fraud patterns that traditional models may miss.

## Conclusion

This project showcases the practical application of Graph Neural Networks in fraud detection, highlighting their advantages over traditional methods. The use of GNNs allows for the capture of complex transactional relationships and provides a robust framework for detecting fraud in real-world datasets.