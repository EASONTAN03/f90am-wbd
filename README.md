World bank data project instruction 

# Set up the python env 
- conda env create -f environment.yml 
-- pytorch need to be manually installed with device preference (cuda/cpu)
- conda avtivate wbd

# Hyperparameters setting and operations.
1. Task 1: Data Exploration and Missing Value Imputation
    + Hyperparameters: None
    + Run src/task1.py

2. Task 2: Dimensionality Reduction and Clustering using Autoencoder
    + Hyperparameters: 
        - seed
        - batch_sizes
        - epochs_list
        - learning_rates
        - dropout_options
        - latent_list
    + Run src/task2.py

3. Task 3: GDP Classification Using MLP
    + Hyperparameters: 
        - seed
        - batch_sizes
        - epochs_list
        - learning_rates
        - scoring_metric
        - k_folds
    + Run src/task3.py

4. Task 4: Time-Series GDP Forecasting Using Deep Learning Models
    + Hyperparameters: 
        - seed
        - models
        - batch_sizes
        - epochs_list
        - learning_rates
        - dropout_options
        - val_size
        - test_size
    + Run src/task4.py

5. Task 5: Variational Autoencoder for Data Augmentation
    + Hyperparameters: 
        - seed
        - batch_size
        - epochs
        - learning_rate
        - hidden_dim
        - latent_dim
        - dataset_timestamp
    + Run src/task5.py

6. Task 6: GDP Forecasting with VAE-Augmented Data
    + Hyperparameters: 
        - seed
        - models
        - batch_sizes
        - epochs_list
        - learning_rates
        - dropout_options
        - hidden_dim
        - latent_dim
        - timestamp_task4
        - timestamp_task5
    + Run src/task6.py
      
Collaborators: Tan Beng Seh and Ng Rou Yan
