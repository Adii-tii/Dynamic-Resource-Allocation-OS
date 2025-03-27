import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import FancyBboxPatch
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import random

data = pd.read_csv('resource_data_labeled.csv')
print("Original data shape:", data.shape)

strategy_cols = ['Throttle', 'Prioritize', 'Reduce_Switching', 'Balance_Load', 
                 'Boost_Parallel', 'Handle_Burst', 'Swap_Out', 'Optimize_Memory', 
                 'Increase_VMS', 'Reduce_VMS', 'Prioritize_RAM']

# Features for augmentation (numerical + categorical)
numerical_cols = ['Total_CPU_%', 'Total_RAM_%', 'Free_RAM_%', 'CPU_Usage_%', 
                  'Context_Switches', 'Memory_Usage_%', 'VMS_MB', 'Thread_Count']
categorical_cols = ['User_Priority']
feature_cols = numerical_cols + categorical_cols

# Encode categorical feature (User_Priority)
label_encoder = LabelEncoder()
data['User_Priority'] = label_encoder.fit_transform(data['User_Priority'])

target_total_rows = 4000

target_all_zero_rows = int(target_total_rows * 0.15)  # ~600 rows (15%)

rows_with_strategy = target_total_rows - target_all_zero_rows  # ~3400 rows
target_positive_per_strategy = rows_with_strategy // len(strategy_cols)  # ~309
target_negative_per_strategy = target_positive_per_strategy  # ~309

# Function to augment data for a single strategy using SMOTE
def augment_strategy(data, strategy, target_positive_count, target_negative_count):
    # Separate features and labels
    X = data[feature_cols]
    y = data[strategy]
    
    # Count current class distribution
    positive_count = (y == 1).sum()
    negative_count = (y == 0).sum()
    print(f"{strategy} - Original: Class 0: {negative_count}, Class 1: {positive_count}")
    
    # Apply SMOTE to oversample the positive class (label 1)
    if positive_count < target_positive_count:
        smote = SMOTE(sampling_strategy={1: target_positive_count}, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y
    
    # Create a DataFrame with resampled data
    resampled_data = pd.DataFrame(X_resampled, columns=feature_cols)
    resampled_data[strategy] = y_resampled
    
    # Adjust positive and negative classes to match targets
    positive_data = resampled_data[resampled_data[strategy] == 1]
    negative_data = resampled_data[resampled_data[strategy] == 0]
    
    # Adjust positive class (label 1)
    if len(positive_data) > target_positive_count:
        positive_data = resample(positive_data, replace=False, n_samples=target_positive_count, random_state=42)
    elif len(positive_data) < target_positive_count:
        positive_data = resample(positive_data, replace=True, n_samples=target_positive_count, random_state=42)
    
    # Adjust negative class (label 0)
    if len(negative_data) > target_negative_count:
        negative_data = resample(negative_data, replace=False, n_samples=target_negative_count, random_state=42)
    elif len(negative_data) < target_negative_count:
        negative_data = resample(negative_data, replace=True, n_samples=target_negative_count, random_state=42)
    
    # Combine positive and negative data
    augmented_data = pd.concat([positive_data, negative_data], ignore_index=True)
    print(f"{strategy} - After augmentation: Class 0: {(augmented_data[strategy] == 0).sum()}, Class 1: {(augmented_data[strategy] == 1).sum()}")
    
    # Add back other columns
    for col in data.columns:
        if col not in feature_cols and col != strategy:
            augmented_data[col] = data[col].sample(n=len(augmented_data), replace=True, random_state=42).values
    
    return augmented_data

# Step 1: Augment each strategy independently for the training dataset
augmented_datasets = []
for strategy in strategy_cols:
    augmented_data = augment_strategy(data, strategy, target_positive_per_strategy, target_negative_per_strategy)
    augmented_datasets.append(augmented_data)

# Step 2: Create a balanced training dataset by sampling rows for each strategy
# We want each strategy to have exactly 309 positive samples in the final dataset
train_data = pd.DataFrame(columns=data.columns)
rows_per_strategy = target_positive_per_strategy  # 309 rows with label 1 per strategy

# Step 3: Sample exactly 309 positive samples for each strategy
for i, strategy in enumerate(strategy_cols):
    augmented_data = augmented_datasets[i]
    
    # Sample exactly 309 positive samples for this strategy
    positive_samples = augmented_data[augmented_data[strategy] == 1]
    positive_sampled = positive_samples.sample(n=rows_per_strategy, replace=True, random_state=42)
    
    # Initialize other strategy columns to 0 for these rows
    for other_strategy in strategy_cols:
        if other_strategy != strategy:
            positive_sampled[other_strategy] = 0
    
    train_data = pd.concat([train_data, positive_sampled], ignore_index=True)

# Step 4: Add negative samples to reach 4000 rows, including all-zero rows
current_rows = len(train_data)  # Should be 309 * 11 = 3399 rows
remaining_rows = target_total_rows - current_rows  # Should be 4000 - 3399 = 601 rows

if remaining_rows > 0:
    # Sample negative samples (rows with all strategies set to 0) from the original data
    negative_samples = data.copy()
    for strategy in strategy_cols:
        negative_samples[strategy] = 0
    negative_sampled = negative_samples.sample(n=remaining_rows, replace=True, random_state=42)
    train_data = pd.concat([train_data, negative_sampled], ignore_index=True)

# Step 5: Adjust all-zero rows to the target number (~600 rows)
all_zeros = (train_data[strategy_cols].sum(axis=1) == 0)
current_all_zero_count = all_zeros.sum()
if current_all_zero_count > target_all_zero_rows:
    # Reduce all-zero rows by assigning strategies to some of them
    all_zero_indices = train_data[all_zeros].index
    excess_all_zeros = current_all_zero_count - target_all_zero_rows
    indices_to_assign = np.random.choice(all_zero_indices, size=excess_all_zeros, replace=False)
    for idx in indices_to_assign:
        strategy_to_assign = np.random.choice(strategy_cols)
        train_data.loc[idx, strategy_to_assign] = 1
elif current_all_zero_count < target_all_zero_rows:
    # Increase all-zero rows by setting some rows to all zeros
    non_zero_rows = train_data[~all_zeros].index
    needed_all_zeros = target_all_zero_rows - current_all_zero_count
    indices_to_zero = np.random.choice(non_zero_rows, size=needed_all_zeros, replace=False)
    for idx in indices_to_zero:
        for strategy in strategy_cols:
            train_data.loc[idx, strategy] = 0

# Step 6: Inverse transform the categorical feature (User_Priority) for the training data
train_data['User_Priority'] = label_encoder.inverse_transform(train_data['User_Priority'].astype(int))

# Step 7: Ensure the training dataset has 4000 rows
print("Training dataset shape:", train_data.shape)

# Step 8: Check all-zero rows in the training dataset
all_zeros = (train_data[strategy_cols].sum(axis=1) == 0).sum()
print(f"Number of all-zero rows in training dataset: {all_zeros} ({all_zeros/len(train_data)*100:.2f}%)")

# Step 9: Save the augmented training dataset
train_data.to_csv('resource_data_train_augmented_4k_smote_balanced.csv', index=False)
print("Augmented training dataset saved to 'resource_data_train_augmented_4k_smote_balanced.csv'")
print("Training label distribution:")
print(train_data[strategy_cols].sum())

# Step 10: Create a balanced test dataset
# For the test set, we want 800 rows per strategy with a 50:50 split (400 class 0, 400 class 1)
test_rows_per_class = 400  # 400 class 0 and 400 class 1 per strategy
test_data = pd.DataFrame(columns=data.columns)

for strategy in strategy_cols:
    # Re-augment the data for the test set with a higher target to ensure enough samples
    augmented_data = augment_strategy(data, strategy, test_rows_per_class, test_rows_per_class)
    
    # Sample 400 positive and 400 negative samples for this strategy
    positive_samples = augmented_data[augmented_data[strategy] == 1]
    negative_samples = augmented_data[augmented_data[strategy] == 0]
    
    positive_sampled = positive_samples.sample(n=test_rows_per_class, replace=True, random_state=42)
    negative_sampled = negative_samples.sample(n=test_rows_per_class, replace=True, random_state=42)
    
    # Initialize other strategy columns to 0 for these rows
    for other_strategy in strategy_cols:
        if other_strategy != strategy:
            positive_sampled[other_strategy] = 0
            negative_sampled[other_strategy] = 0
    
    # Combine positive and negative samples
    strategy_test_data = pd.concat([positive_sampled, negative_sampled], ignore_index=True)
    test_data = pd.concat([test_data, strategy_test_data], ignore_index=True)

# Step 11: Adjust the test dataset size to exactly 800 rows per strategy (8800 rows total)
# Since we have 11 strategies, we expect 11 * 800 = 8800 rows
expected_test_rows = len(strategy_cols) * (test_rows_per_class * 2)  # 8800 rows
if len(test_data) > expected_test_rows:
    test_data = test_data.sample(n=expected_test_rows, random_state=42)
elif len(test_data) < expected_test_rows:
    additional_rows = expected_test_rows - len(test_data)
    extra_data = test_data.sample(n=additional_rows, replace=True, random_state=42)
    test_data = pd.concat([test_data, extra_data], ignore_index=True)

# Step 12: Inverse transform the categorical feature (User_Priority) for the test data
test_data['User_Priority'] = label_encoder.inverse_transform(test_data['User_Priority'].astype(int))

# Step 13: Ensure the test dataset has the correct number of rows
print("Test dataset shape:", test_data.shape)

# Step 14: Check all-zero rows in the test dataset
all_zeros = (test_data[strategy_cols].sum(axis=1) == 0).sum()
print(f"Number of all-zero rows in test dataset: {all_zeros} ({all_zeros/len(test_data)*100:.2f}%)")

# Step 15: Save the augmented test dataset
test_data.to_csv('resource_data_augmented_new3.csv', index=False)
print("Augmented test dataset saved to 'resource_data_test_augmented_balanced.csv'")
print("Test label distribution:")
print(test_data[strategy_cols].sum())
