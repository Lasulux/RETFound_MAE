import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/metadata.csv")
normal_df = df[df["N"] == 1]  # both eyes are healthy
normal_left_df = normal_df[["ID", "Patient Age", "Patient Sex", "Left-Fundus"]]

# Create stratified train and test sets
stratify_columns = normal_left_df[['Patient Age', 'Patient Sex']]
train_df, test_df = train_test_split(normal_left_df, test_size=0.2, stratify=stratify_columns)

#%% Plot patient age distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot age distribution for training data
sns.histplot(train_df['Patient Age'], kde=True, color='blue', stat='density', ax=axes[0])
axes[0].set_title('Age Distribution - Train Set')
axes[0].set_xlabel('Patient Age')
axes[0].set_ylabel('Density')

# Plot age distribution for testing data
sns.histplot(test_df['Patient Age'], kde=True, color='red', stat='density', ax=axes[1])
axes[1].set_title('Age Distribution - Test Set')
axes[1].set_xlabel('Patient Age')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.show()

# Plot patient sex distributions
# Combine the DataFrames for plotting
train_df['Dataset'] = 'Train'
test_df['Dataset'] = 'Test'
combined_df = pd.concat([train_df, test_df], ignore_index=True)

plt.figure(figsize=(8, 6))
sns.countplot(data=combined_df, x='Patient Sex', hue='Dataset')
plt.title('Sex Distribution')
plt.xlabel('Patient Sex')
plt.ylabel('Density')

# Show the plot
plt.show()