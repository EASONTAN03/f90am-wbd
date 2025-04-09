import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from tqdm import tqdm  # Progress bar

data_path = r"data/world_bank_data_dev.csv"
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"]).dt.year


from sklearn.impute import KNNImputer
def impute_knn(df):
    """Impute missing values using KNN with data from all countries."""
    imputer = KNNImputer(n_neighbors=3)
    numeric_df = df.select_dtypes(include=[np.number])
    imputed_data = imputer.fit_transform(numeric_df)
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)
    for col in df.columns:
        if col not in imputed_df.columns:
            imputed_df[col] = df[col]
    return imputed_df

def identify_missing_sequences(matrix):
    """ Identify missing sequences in each column of a matrix """
    is_nan = np.isnan(matrix)  # Boolean mask for missing values
    missing_indices = np.zeros_like(matrix, dtype=int)  # Store missing sequence indices
    # Compute missing groups (vectorized)
    for col in range(matrix.shape[1]):
        series=matrix[:, col]
        series = np.asarray(series)  # Convert to NumPy array
        is_nan = np.isnan(series)  # Identify missing values (NaN)
        
        # Fix: Ensure the first NaN is also counted
        missing_groups = np.cumsum(is_nan & np.concatenate(([True], ~is_nan[:-1])))  # Fix for first NaN
        missing_indices[:, col] = is_nan * missing_groups  # Assign sequence numbers

    return missing_indices

def auto_arima_impute(series, inverse=False):
    """ Auto ARIMA-based imputation for missing sequences """
    if not inverse:
        series = pd.Series(series).sort_values(ascending=True)
        valid_data = series.dropna()

        # Fit ARIMA on valid values
        model = auto_arima(valid_data, seasonal=False, suppress_warnings=True)

        # Predict missing values
        missing_idx = np.where(series.isna())[0]

        # print(series)
        # print(np.where(series.isna()))
        # print(missing_idx)
        predictions = model.predict(n_periods=len(missing_idx))
        predictions = np.flip(predictions)

    if inverse:
        series = pd.Series(series)
        valid_data = series.dropna()

        # Fit ARIMA on valid values
        valid_data = np.log1p(valid_data)
        model = auto_arima(valid_data, seasonal=False, suppress_warnings=True)

        # Predict missing values
        missing_idx = np.where(series.isna())[0]

        # print(series)
        # print(np.where(series.isna()))
        # print(missing_idx)
        predictions = model.predict(n_periods=len(missing_idx))
        predictions = np.expm1(model.predict(n_periods=len(missing_idx)))

    return np.array(predictions)

def impute_missing_values(matrix):
    """ Main function for matrix-based imputation """
    matrix = np.array(matrix, dtype=float)  # Convert to NumPy array
    missing_indices = identify_missing_sequences(matrix)
    
    for col in range(matrix.shape[1]):
        col_data = matrix[:, col]
        missing_seq = np.unique(missing_indices[:, col])
        for seq in missing_seq:
            is_nan=np.isnan(col_data)
            if seq == 0:
                continue  # Skip if not missing
            
            indices = np.where(missing_indices[:, col] == seq)[0]
            start, end, length = indices[0], indices[-1], len(indices)
            if length == 1:
                if start == 0:
                    col_data[start] = col_data[start+1]
                elif end == 43:
                    col_data[end] = col_data[end-1]
                else:
                    col_data[start] = (col_data[start+1]+col_data[start-1])/2
            elif length == 2 and (start-1) >=0 and (end+1) <44:
                col_data[start] = col_data[start-1]
                col_data[end] = col_data[end+1]
            elif length<15 and start == 0 and np.sum(is_nan[end+1:end+21])==0:
                arima_pred = auto_arima_impute(col_data[start:end+21].flatten())
                col_data[start:end+1] = arima_pred
            elif length<15 and end == 43 and np.sum(is_nan[start-15:start])==0:
                arima_pred = auto_arima_impute(col_data[start-15:end+1].flatten(), inverse=True)
                col_data[start:end+1] = arima_pred
            elif length != 44:
                try:
                    if (end+1) <=43:
                            col_data[start:end+1] = col_data[end + 1]
                    elif (start-1) >=0:
                        col_data[start:end+1] = col_data[start - 1]
                except Exception:
                    try:
                        col_data[start:end+1] = np.mean(col_data[~np.isnan(col_data)]) 
                    except Exception:
                        continue
            
        matrix[:, col] = col_data

    return matrix

processed_df_list = []
for country in tqdm(df['country'].unique(), desc="Processing Countries", unit="country"):
    # print(f"Processing {country}...")
    country_df = df[df['country'] == country].copy()
    country_labels = country_df[["country"]]
    numerical_columns = country_df.drop(columns=["country","date"])

    imputed_matrix = impute_missing_values(numerical_columns)
    imputed_df = pd.DataFrame(imputed_matrix, columns=numerical_columns.columns)
    imputed_df = pd.concat([country_labels.reset_index(drop=True), imputed_df], axis=1)

    processed_df_list.append(imputed_df)

output_imputed_df = pd.concat(processed_df_list, ignore_index=True)

# Retrieve original date values for each row using country as key
date_mapping = np.array(df['date'])

# Convert date to integer (if it's not already)
output_imputed_df['date'] = date_mapping

# Rearrange columns: Move 'date' to the second position
cols = output_imputed_df.columns.tolist()
cols.remove('date')  # Remove 'date' from its current position
cols.insert(1, 'date')  # Insert 'date' as the second column
output_imputed_df = output_imputed_df[cols]  # Reorder dataframe

final_imputed_df = impute_knn(output_imputed_df)
final_imputed_df = final_imputed_df.reindex(columns=['country'] + [col for col in final_imputed_df.columns if col != 'country'])
final_imputed_df["Unemployment_rate"] = pd.to_numeric(final_imputed_df["Unemployment_rate"], errors='coerce')
final_imputed_df["Unemployment_rate"] = final_imputed_df["Unemployment_rate"].clip(lower=0)
final_imputed_df.to_csv(r"data/final_impute_world_bank_data_dev.csv",index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

# Convert date column to datetime and sort
def normalize_and_plot(df, filter_countries=True, save_fig=False, file_dir=None):
    if save_fig:
        output_dir="results"
        os.makedirs(output_dir, exist_ok=True)
    
    df_plot=df.copy()
    df_plot = df_plot.sort_values(["country", "date"])  # Ensure chronological order

    # Select relevant countries
    if filter_countries==True:
        countries = ["United States", "China", "Russian Federation", "Brazil"]
        filtered_df = df_plot[df_plot["country"].isin(countries)].copy()
    else:
        countries = list(sorted(df_plot['country'].unique()))
        filtered_df = df_plot.copy()
        
    # Define indicators for different scaling methods

    standard_indicators = [
        "Life_expectancy", "Literacy_rate", "Unemployment_rate",
        "Fertility_rate", "Poverty_ratio", "Primary_school_enrolment_rate",
        "GDPpc_2017$", "Population_total", "Energy_use", "Exports_2017$"
    ]

    log_transform_indicators = [
    ]

    # Create a new DataFrame for transformed data
    normalized_df = filtered_df.copy()

    scaler = StandardScaler()
    for indicator in standard_indicators:
        normalized_df[indicator]=scaler.fit_transform(filtered_df[[indicator]])

    ### **2️⃣ Apply Log Transformation for selected indicators**
    for indicator in log_transform_indicators:
        normalized_df[indicator] = np.log1p(normalized_df[indicator])  # log1p to avoid log(0) issues

    # Plot each indicator for all countries
    sns.set_style("whitegrid")

    for i,indicator in enumerate(standard_indicators + log_transform_indicators):
        plt.figure(figsize=(12, 6))
        for country in countries:
            subset = normalized_df[normalized_df["country"] == country]
            # Plot only available (non-null) points with "-o"
            plt.plot(subset["date"], subset[indicator], "-o", label=country)

        plt.title(f"{indicator} Over Time")
        plt.xlabel("Year")
        plt.ylabel("Scaled Value" if indicator in standard_indicators else "Log-Transformed Value")
        plt.legend()

        if save_fig:
            # Save plot as numbered PNG
            os.makedirs(f"{output_dir}/{file_dir}",exist_ok=True)
            filename = f"{output_dir}/{file_dir}/{i}_{indicator.replace('$', '').replace(' ', '_')}.png"
            plt.savefig(filename,dpi=300)
            plt.close()  # Close figure to free memory
        else:
            continue

    return normalized_df

imputed_df=pd.read_csv(r"data\final_impute_world_bank_data_dev.csv")
print(imputed_df.isnull().sum())
normalised_imputed_df=normalize_and_plot(imputed_df, filter_countries=True, save_fig=True, file_dir="task1")
normalised_imputed_df=normalize_and_plot(imputed_df, filter_countries=False, save_fig=False, file_dir="task1")
normalised_imputed_df.to_csv('data/normalised_imputed_world_bank_data_dev.csv', index=False)

normalised_df=pd.read_csv('data/normalised_imputed_world_bank_data_dev.csv')

import pandas as pd
import numpy as np

def create_sequences(df, window_size=5, step_size=1):
    sequences = []
    countries = df['country'].unique()
    
    for country in countries:
        country_df = df[df['country'] == country].sort_values(by='date')
        indicators = country_df.columns.difference(['country', 'date'])
        
        for start in range(0, len(country_df) - window_size + 1, step_size):
            end = start + window_size
            sequence = country_df.iloc[start:end][indicators].values.flatten()
            
            if len(sequence) == window_size * len(indicators):  # Ensure proper sequence length
                formatted_sequence = ", ".join(map(str, sequence))  # Convert to comma-separated string
                sequences.append((country, formatted_sequence))
    
    return sequences

window_size=5

# Assuming 'final_imputed_df' is your DataFrame with the imputed data
sequences = create_sequences(normalised_df,window_size=window_size)

# Convert the sequences to a DataFrame for better visualization
sequences_df = pd.DataFrame(sequences, columns=['country', 'sequence'])

# Display the first few sequences
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_colwidth', None)  # Display full content of each column
print(sequences_df.head(5))

# Save to CSV
sequences_df.to_csv(f'data/task2_world_bank_data_dev.csv', index=False)