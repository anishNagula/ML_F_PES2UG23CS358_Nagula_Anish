# numpy_sklearn.py
import numpy as np
import pandas as pd
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    if data.shape[0] == 0:
        return 0.0

    target_column = data[:, -1]
    _, counts = np.unique(target_column, return_counts=True)
    
    probabilities = counts / len(target_column)
    
    # Add a small epsilon to avoid log2(0)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    
    return entropy


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    """
    if data.shape[0] == 0:
        return 0.0

    attribute_column = data[:, attribute]
    unique_values = np.unique(attribute_column)
    
    total_samples = len(attribute_column)
    avg_info = 0.0
    
    for value in unique_values:
        # Create a subset of the data where the attribute has the current value
        subset = data[data[:, attribute] == value]
        
        # Calculate the weight (proportion of samples with this value)
        weight = subset.shape[0] / total_samples
        
        # Calculate the entropy of the subset
        subset_entropy = get_entropy_of_dataset(subset)
        
        # Add the weighted entropy to the average information
        avg_info += weight * subset_entropy
        
    return avg_info


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    """
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info_attribute = get_avg_info_of_attribute(data, attribute)
    information_gain = dataset_entropy - avg_info_attribute
    
    return round(information_gain, 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    """
    num_attributes = data.shape[1] - 1
    information_gains = {}
    
    for i in range(num_attributes):
        information_gains[i] = get_information_gain(data, i)
        
    if not information_gains:
        return {}, -1

    selected_attribute = max(information_gains, key=information_gains.get)
    
    return information_gains, selected_attribute

# --- Helper function to load and preprocess data ---
def load_and_preprocess_data(file_path):
    """Loads a CSV, handles dataset-specific quirks, and converts to a NumPy array."""
    df = pd.read_csv(file_path, header=None)
    
    # Handle specific dataset properties
    if "mushroom" in file_path:
        df.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
                      'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
                      'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                      'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 
                      'veil-type', 'veil-color', 'ring-number', 'ring-type', 
                      'spore-print-color', 'population', 'habitat']
        target = df.pop('class')
        df['class'] = target
        df = df.drop('veil-type', axis=1)
    elif "tictactoe" in file_path:
        df.columns = ['top-left', 'top-middle', 'top-right', 'middle-left', 'middle-middle',
                      'middle-right', 'bottom-left', 'bottom-middle', 'bottom-right', 'Class']
    elif "nursery" in file_path:
        df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 
                      'social', 'health', 'class']

    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # Label encode all columns
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    # Convert to NumPy array
    return df.values

# --- Main execution block ---
if __name__ == '__main__':
    # --- CHOOSE YOUR DATASET ---
    # Just change the file_path to 'mushroom.csv', 'tictactoe.csv', or 'nursery.csv'
    file_path = 'tictactoe.csv'  # <-- CHANGE THIS LINE
    
    print(f"--- Analyzing {file_path} with NumPy ---")
    
    # 1. Load data
    dataset_array = load_and_preprocess_data(file_path)
    print(f"Loaded array shape: {dataset_array.shape}")
    
    # 2. Test the functions
    entropy = get_entropy_of_dataset(dataset_array)
    print(f"Dataset Entropy: {entropy:.4f}")
    
    # Test for the first attribute (index 0)
    avg_info = get_avg_info_of_attribute(dataset_array, 0)
    print(f"Avg Info of Attribute 0: {avg_info:.4f}")
    
    info_gain = get_information_gain(dataset_array, 0)
    print(f"Info Gain of Attribute 0: {info_gain:.4f}")
    
    # 3. Get the best attribute
    gains, best_attr = get_selected_attribute(dataset_array)
    print(f"\nInformation Gains for all attributes:\n{gains}")
    print(f"\nBest attribute to split on: Index {best_attr} (with gain: {gains[best_attr]:.4f})")
