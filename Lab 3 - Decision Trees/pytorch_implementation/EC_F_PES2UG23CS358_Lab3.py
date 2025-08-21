# pytorch.py
import torch
import pandas as pd

def get_entropy_of_dataset(tensor: torch.Tensor) -> float:
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    if tensor.shape[0] == 0:
        return 0.0

    target_column = tensor[:, -1]
    # Get unique values and their counts
    unique_classes, counts = torch.unique(target_column, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts.float() / len(target_column)
    
    # Calculate entropy
    # We add a small epsilon to avoid log2(0)
    entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-9))
    
    return entropy.item()


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    if tensor.shape[0] == 0:
        return 0.0

    attribute_column = tensor[:, attribute]
    unique_values = torch.unique(attribute_column)
    
    total_samples = len(attribute_column)
    avg_info = 0.0
    
    for value in unique_values:
        # Create a subset of the data where the attribute has the current value
        subset_mask = tensor[:, attribute] == value
        subset = tensor[subset_mask]
        
        # Calculate the weight (proportion of samples with this value)
        weight = subset.shape[0] / total_samples
        
        # Calculate the entropy of the subset
        subset_entropy = get_entropy_of_dataset(subset)
        
        # Add the weighted entropy to the average information
        avg_info += weight * subset_entropy
        
    return avg_info


def get_information_gain(tensor: torch.Tensor, attribute: int) -> float:
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info_attribute = get_avg_info_of_attribute(tensor, attribute)
    information_gain = dataset_entropy - avg_info_attribute
    
    return round(information_gain, 4)


def get_selected_attribute(tensor: torch.Tensor) -> tuple:
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    num_attributes = tensor.shape[1] - 1
    information_gains = {}
    
    for i in range(num_attributes):
        information_gains[i] = get_information_gain(tensor, i)
        
    if not information_gains:
        return {}, -1

    selected_attribute = max(information_gains, key=information_gains.get)
    
    return information_gains, selected_attribute

# --- Helper function to manually label encode a column ---
def custom_label_encoder(column):
    """
    Encodes a pandas Series with categorical values into integer labels.
    
    Args:
        column (pd.Series): The column to encode.
        
    Returns:
        pd.Series: The column with encoded integer labels.
    """
    unique_values = column.unique()
    mapping = {value: i for i, value in enumerate(unique_values)}
    return column.map(mapping)

# --- Helper function to load and preprocess data ---
def load_and_preprocess_data(file_path):
    """Loads a CSV, handles dataset-specific quirks, and converts to a PyTorch tensor."""
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
    
    # Label encode all columns using the custom function
    for col in df.columns:
        df[col] = custom_label_encoder(df[col])
        
    # Convert to PyTorch tensor
    return torch.tensor(df.values, dtype=torch.float32)

# --- Main execution block ---
if __name__ == '__main__':
    # --- CHOOSE YOUR DATASET ---
    # Just change the file_path to 'mushroom.csv', 'tictactoe.csv', or 'nursery.csv'
    file_path = 'mushroom.csv'  # <-- CHANGE THIS LINE
    
    print(f"--- Analyzing {file_path} with PyTorch ---")
    
    # 1. Load data
    dataset_tensor = load_and_preprocess_data(file_path)
    print(f"Loaded tensor shape: {dataset_tensor.shape}")
    
    # 2. Test the functions
    entropy = get_entropy_of_dataset(dataset_tensor)
    print(f"Dataset Entropy: {entropy:.4f}")
    
    # Test for the first attribute (index 0)
    avg_info = get_avg_info_of_attribute(dataset_tensor, 0)
    print(f"Avg Info of Attribute 0: {avg_info:.4f}")
    
    info_gain = get_information_gain(dataset_tensor, 0)
    print(f"Info Gain of Attribute 0: {info_gain:.4f}")
    
    # 3. Get the best attribute
    gains, best_attr = get_selected_attribute(dataset_tensor)
    print(f"\nInformation Gains for all attributes:\n{gains}")
    print(f"\nBest attribute to split on: Index {best_attr} (with gain: {gains[best_attr]:.4f})")
