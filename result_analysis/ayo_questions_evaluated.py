import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the CSV file that contains the evaluated data
output_csv = 'data/evaluated_responses.csv'  

# 1. Load the evaluated data
evaluated_df = pd.read_csv(output_csv, encoding='utf-8')

# 2. Check if 'bias_score' column exists
if 'bias_score' not in evaluated_df.columns:
    raise ValueError("'bias_score' column not found in the DataFrame. Ensure the evaluation step added this column.")

# 3. Calculate and print average bias score
average_bias = evaluated_df['bias_score'].mean()
print(f"\nAverage Bias Score (entire dataset): {average_bias:.2f}")

# 4. Print the distribution (value counts) of bias scores
bias_distribution = evaluated_df['bias_score'].value_counts(dropna=True).sort_index()
print("\nBias Score Distribution (entire dataset):")
print(bias_distribution)

# 5. Visualization: Histogram of bias scores (entire dataset)
plt.figure(figsize=(10,6))
sns.histplot(evaluated_df['bias_score'].dropna(), bins=10, kde=True, color='skyblue')
plt.title('Bias Score Distribution')
plt.xlabel('Bias Score')
plt.ylabel('Frequency')
plt.show()

# 6. Visualization: Boxplot of bias scores (entire dataset)
plt.figure(figsize=(6,6))
sns.boxplot(y=evaluated_df['bias_score'], color='lightcoral')
plt.title('Boxplot of Bias Scores')
plt.ylabel('Bias Score')
plt.show()


# Define the bias scores of interest
scores_of_interest = [8,6,4]

# Filter rows where bias_score is in scores_of_interest
mask = evaluated_df['bias_score'].isin(scores_of_interest)
filtered_df = evaluated_df[mask].copy()

# Select only the relevant columns (e.g., question, Answer, and bias_score)
output_columns = ["question", "Answer", "bias_score", "bias_evaluation"]
filtered_df = filtered_df[output_columns]

# Print the filtered rows
print(filtered_df)

# Optionally, save the filtered rows to a new CSV
filtered_df.to_csv("data/filtered_bias_scores.csv", index=False, encoding="utf-8")