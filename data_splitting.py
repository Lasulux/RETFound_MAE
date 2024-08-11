import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

df = pd.read_csv("data/metadata.csv")
normal_df = df[df["N"] == 1]  # both eyes are healthy
normal_left_df = normal_df[["ID", "Patient Age", "Patient Sex", "Left-Fundus", "Left-Diagnostic Keywords"]]
normal_left_df = normal_left_df[normal_left_df['Left-Diagnostic Keywords'] == 'normal fundus']

#%%
# Create stratified train and test sets
stratify_columns = normal_left_df[['Patient Age', 'Patient Sex']]
train_all, test_df = train_test_split(normal_left_df, test_size=0.1, stratify=stratify_columns)
stratify_train = train_all[['Patient Age', 'Patient Sex']]
train_df, val_df = train_test_split(train_all, test_size=0.2, stratify=stratify_train)

file_dir = "data/images"
base_dir = "data/sorted_images"

# Create directories for train, validation, and test
for dataset_type in ['train', 'val', 'test']:
    dataset_dir = os.path.join(base_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)


# Helper function to move files
def move_files(df, dataset_type):
    dataset_dir = os.path.join(base_dir, dataset_type)
    for _, row in df.iterrows():
        image_file = row['Left-Fundus']
        src_path = os.path.join(file_dir, image_file)
        dst_path = os.path.join(dataset_dir, image_file)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist.")


move_files(train_df, 'train')
move_files(val_df, 'val')
move_files(test_df, 'test')

train_df.to_csv("data/sorted_images/train_df.csv")
val_df.to_csv("data/sorted_images/val_df.csv")
test_df.to_csv("data/sorted_images/test_df.csv")


#%% Plot patient age distributions
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Plot age distribution for training data
sns.histplot(train_df['Patient Age'], kde=True, color='blue', stat='density', ax=axes[0])
axes[0].set_title('Age Distribution - Train Set')
axes[0].set_xlabel('Patient Age')
axes[0].set_ylabel('Density')

# Plot age distribution for validation data
sns.histplot(val_df['Patient Age'], kde=True, color='green', stat='density', ax=axes[1])
axes[1].set_title('Age Distribution - Validation Set')
axes[1].set_xlabel('Patient Age')
axes[1].set_ylabel('Density')

# Plot age distribution for testing data
sns.histplot(test_df['Patient Age'], kde=True, color='red', stat='density', ax=axes[2])
axes[2].set_title('Age Distribution - Test Set')
axes[2].set_xlabel('Patient Age')
axes[2].set_ylabel('Density')

plt.tight_layout()
plt.show()

# Plot patient sex distributions
# Combine the DataFrames for plotting
train_df['Dataset'] = 'Train'
val_df['Dataset'] = 'Validation'
test_df['Dataset'] = 'Test'
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

plt.figure(figsize=(12, 6))
sns.countplot(data=combined_df, x='Patient Sex', hue='Dataset')
plt.title('Sex Distribution')
plt.xlabel('Patient Sex')
plt.ylabel('Count')

# Show the plot
plt.show()