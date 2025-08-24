# /*
# import numpy as np
# from collections import Counter

# def get_entropy_of_dataset(data: np.ndarray) -> float:
#     """
#     Calculate the entropy of the entire dataset using the target variable (last column).
    
#     Args:
#         data (np.ndarray): Dataset where the last column is the target variable
    
#     Returns:
#         float: Entropy value calculated using the formula: 
#                Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    
#     Example:
#         data = np.array([[1, 0, 'yes'],
#                         [1, 1, 'no'],
#                         [0, 0, 'yes']])
#         entropy = get_entropy_of_dataset(data)
#         # Should return entropy based on target column ['yes', 'no', 'yes']
#     """
#     # TODO: Implement entropy calculation
#     # Hint: Use np.unique() to get unique classes and their counts
#     # Hint: Handle the case when probability is 0 to avoid log2(0)
#     pass


# def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
#     """
#     Calculate the average information (weighted entropy) of a specific attribute.
    
#     Args:
#         data (np.ndarray): Dataset where the last column is the target variable
#         attribute (int): Index of the attribute column to calculate average information for
    
#     Returns:
#         float: Average information calculated using the formula:
#                Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
#                where S_v is subset of data with attribute value v
    
#     Example:
#         data = np.array([[1, 0, 'yes'],
#                         [1, 1, 'no'],
#                         [0, 0, 'yes']])
#         avg_info = get_avg_info_of_attribute(data, 0)  # For attribute at index 0
#         # Should return weighted average entropy for attribute splits
#     """
#     # TODO: Implement average information calculation
#     # Hint: For each unique value in the attribute column:
#     #   1. Create a subset of data with that value
#     #   2. Calculate the entropy of that subset
#     #   3. Weight it by the proportion of samples with that value
#     #   4. Sum all weighted entropies
#     pass


# def get_information_gain(data: np.ndarray, attribute: int) -> float:
#     """
#     Calculate the Information Gain for a specific attribute.
    
#     Args:
#         data (np.ndarray): Dataset where the last column is the target variable
#         attribute (int): Index of the attribute column to calculate information gain for
    
#     Returns:
#         float: Information gain calculated using the formula:
#                Information_Gain = Entropy(S) - Avg_Info(attribute)
#                Rounded to 4 decimal places
    
#     Example:
#         data = np.array([[1, 0, 'yes'],
#                         [1, 1, 'no'],
#                         [0, 0, 'yes']])
#         gain = get_information_gain(data, 0)  # For attribute at index 0
#         # Should return the information gain for splitting on attribute 0
#     """
#     # TODO: Implement information gain calculation
#     # Hint: Information Gain = Dataset Entropy - Average Information of Attribute
#     # Hint: Use the functions you implemented above
#     # Hint: Round the result to 4 decimal places
#     pass

# def get_selected_attribute(data: np.ndarray) -> tuple:
#     """
#     Select the best attribute based on highest information gain.
    
#     Args:
#         data (np.ndarray): Dataset where the last column is the target variable
    
#     Returns:
#         tuple: A tuple containing:
#             - dict: Dictionary mapping attribute indices to their information gains
#             - int: Index of the attribute with the highest information gain
    
#     Example:
#         data = np.array([[1, 0, 2, 'yes'],
#                         [1, 1, 1, 'no'],
#                         [0, 0, 2, 'yes']])
#         result = get_selected_attribute(data)
#         # Should return something like: ({0: 0.123, 1: 0.456, 2: 0.789}, 2)
#         # where 2 is the index of the attribute with highest gain
#     """
#     # TODO: Implement attribute selection
#     # Hint: Calculate information gain for all attributes (except target variable)
#     # Hint: Store gains in a dictionary with attribute index as key
#     # Hint: Find the attribute with maximum gain using max() with key parameter
#     # Hint: Return tuple (gain_dictionary, selected_attribute_index)
#     pass

import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the dataset using the target column (last column).
    """

    # Extract the target column (last column in the dataset).
    # Example: if data = [[1,0,'yes'], [0,1,'no']], then y = ['yes','no'].
    y = data[:, -1]

    # Count unique classes and how many times they appear.
    # np.unique(y, return_counts=True) returns (array of classes, array of counts).
    classes, counts = np.unique(y, return_counts=True)

    # Convert counts to probabilities p_i = count/total.
    probabilities = counts / len(y)

    # Apply entropy formula: -Σ p * log2(p).
    # We use vectorized NumPy operations for efficiency.
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the weighted average entropy for a given attribute.
    """

    # Get the target labels (last column).
    y = data[:, -1]

    # Extract the column of the chosen attribute.
    A = data[:, attribute]

    # Find unique values of this attribute and their counts.
    values, counts = np.unique(A, return_counts=True)

    total_samples = len(y)  # total rows
    avg_info = 0.0          # accumulator

    # Iterate through each unique value of the attribute
    for v, cnt in zip(values, counts):
        # Create mask: True where A == v
        mask = (A == v)

        # Subset of labels where attribute == v
        y_subset = y[mask]

        # Calculate entropy of this subset
        subset_entropy = get_entropy_of_dataset(data[mask])

        # Weight by fraction of samples belonging to this branch
        weight = cnt / total_samples

        # Add to the weighted sum
        avg_info += weight * subset_entropy

    return float(avg_info)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate Information Gain for splitting on a given attribute.
    """

    # Entropy of the parent dataset (before split).
    dataset_entropy = get_entropy_of_dataset(data)

    # Expected entropy after splitting on this attribute.
    avg_info = get_avg_info_of_attribute(data, attribute)

    # Information Gain = reduction in entropy.
    ig = dataset_entropy - avg_info

    # Round to 4 decimal places as per lab spec.
    return round(float(ig), 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Compute IG for all attributes and return the dictionary and the best attribute.
    """

    # Number of columns in dataset
    n_cols = data.shape[1]

    # Last column is target, so features go from 0 .. n_cols-2
    n_features = n_cols - 1

    # Dictionary mapping attribute -> IG value
    gain_dict = {}

    for attr in range(n_features):
        gain_dict[attr] = get_information_gain(data, attr)

    # Select the attribute with the highest IG
    # Tie-breaking: lowest attribute index wins (default behavior of max with key).
    selected_attr = max(gain_dict, key=gain_dict.get)

    return gain_dict, selected_attr
