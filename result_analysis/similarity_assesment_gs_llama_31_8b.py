import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

##################################
# 1. Load the Two Datasets
##################################
# A) Gold Standard
df_gold = pd.read_csv("data/Ayo_FAQ_gender_labels.csv", sep=';', encoding='utf-8')

# B) Framework answers
df_framework = pd.read_csv("data\gender_specific_questions_llama_31_8b.csv", encoding='utf-8')

##################################
# 2. Preprocessing / Splitting
##################################
# If "Question-Variants-English" has multiple sub-questions separated by '\n',
# you may want to explode them. If not, skip.
df_gold['Question List'] = df_gold['Question-Variants-English'].astype(str).str.split('\n')
df_gold = df_gold.explode('Question List').reset_index(drop=True)
df_gold.rename(columns={'Question List': 'question_gold'}, inplace=True)

# Clean up whitespace, etc.
df_gold['question_gold'] = df_gold['question_gold'].fillna('').str.strip()
df_gold['Answer'] = df_gold['Answer'].fillna('').str.strip()

df_framework['question'] = df_framework['question'].fillna('').str.strip()
df_framework['answer_framework1'] = df_framework['answer_framework1'].fillna('').str.strip()

# Drop empty rows if needed
df_gold = df_gold[df_gold['question_gold'] != ''].reset_index(drop=True)


##################################
# 3. Load a Sentence-BERT Model
##################################
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

##################################
# 4. Compute Embeddings
##################################
# For gold questions
gold_questions = df_gold['question_gold'].tolist()
gold_embeddings = model.encode(gold_questions, convert_to_tensor=True)

# For framework questions
fw_questions = df_framework['question'].tolist()
fw_q_embeddings = model.encode(fw_questions, convert_to_tensor=True)

##################################
# 5. Match Each Framework Question
##################################
# We'll create a list to store matched indices and scores
matched_indices = []
question_sim_scores = []

for i, fw_emb in enumerate(fw_q_embeddings):
    # Compute similarity to all gold questions
    sim = util.cos_sim(fw_emb, gold_embeddings)  # shape (1, #gold)
    best_idx = sim.argmax().item()               # index of best match
    best_score = sim[0][best_idx].item()         # best similarity score
    matched_indices.append(best_idx)
    question_sim_scores.append(best_score)

df_framework['matched_gold_index'] = matched_indices
df_framework['question_similarity'] = question_sim_scores

print(df_framework)

##################################
# 6. Merge Gold Answer and
#    Compute Answer Similarity
##################################
# We'll retrieve the gold standard answer from df_gold
# for each matched_gold_index
gold_answers = []
gold_questions_matched = []

for idx in range(len(df_framework)):
    gold_idx = df_framework.loc[idx, 'matched_gold_index']
    gold_answers.append(df_gold.loc[gold_idx, 'Answer'])
    gold_questions_matched.append(df_gold.loc[gold_idx, 'question_gold'])

df_framework['matched_question_gold'] = gold_questions_matched
df_framework['gold_standard_answer'] = gold_answers

# Now measure answer similarity
# We'll encode the framework answers and matched gold answers
fw_answer_embeddings = model.encode(df_framework['answer_framework1'].tolist(), convert_to_tensor=True)
gold_answer_embeddings = model.encode(df_framework['gold_standard_answer'].tolist(), convert_to_tensor=True)

answer_sim_scores = util.cos_sim(fw_answer_embeddings, gold_answer_embeddings)
answer_sim_list = []

for i in range(len(df_framework)):
    answer_sim_list.append(answer_sim_scores[i][i].item())

df_framework['answer_similarity'] = answer_sim_list


##################################
# 7. Final Output
##################################
# We'll keep columns of interest
final_cols = [
    'question',               # original framework question
    'answer_framework1',      # framework answer
    'matched_question_gold',  # matched gold question
    'gold_standard_answer',   # matched gold answer
    'question_similarity',    # similarity between questions
    'answer_similarity'       # similarity between answers
]
df_final = df_framework[final_cols]

print(df_final)

# Save to CSV
df_final.to_csv("data/framework_alignment_gs_llama_31_8b.csv", index=False, encoding='utf-8')

print("Alignment complete. Results saved to 'data/framework_gold_alignment.csv'.")

# Histogramms of similaty scores
# 1A. Distribution of Question Similarity
plt.figure(figsize=(8,5))
sns.histplot(df_final['question_similarity'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Distribution of Question Similarity')
plt.xlabel('Question Similarity (0–1)')
plt.ylabel('Frequency')
plt.show()

# 1B. Distribution of Answer Similarity
plt.figure(figsize=(8,5))
sns.histplot(df_final['answer_similarity'].dropna(), bins=20, kde=True, color='lightgreen')
plt.title('Distribution of Answer Similarity')
plt.xlabel('Answer Similarity (0–1)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot: Question vs. Answer Similarity
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_final, 
    x='question_similarity', 
    y='answer_similarity', 
    color='orchid'
)
plt.title('Question Similarity vs. Answer Similarity')
plt.xlabel('Question Similarity')
plt.ylabel('Answer Similarity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

# Correlation and Joint Plot
corr = df_final['question_similarity'].corr(df_final['answer_similarity'])
print(f"Correlation between question_similarity and answer_similarity: {corr:.3f}")

# Joint Plot
sns.jointplot(
    data=df_final, 
    x='question_similarity', 
    y='answer_similarity', 
    kind='reg', 
    height=6
)
plt.show()

# Tabular Sumary
print("Summary Statistics for Question Similarity:")
print(df_final['question_similarity'].describe())

print("\nSummary Statistics for Answer Similarity:")
print(df_final['answer_similarity'].describe())

# Boxplot of similarities
plt.figure(figsize=(8,5))
melted = pd.melt(
    df_final,
    value_vars=['question_similarity','answer_similarity'],
    var_name='similarity_type', 
    value_name='similarity_score'
)
sns.boxplot(data=melted, x='similarity_type', y='similarity_score', palette='Set2')
plt.title('Box Plot of Question vs. Answer Similarity')
plt.xlabel('Similarity Type')
plt.ylabel('Similarity Score')
plt.ylim(0,1)
plt.show()