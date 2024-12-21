import pandas as pd
from bert_score import score
import torch

# 1. Load Your DataFrame
# Example: df_final contains columns "answer_framework1" and "gold_standard_answer"
df_final = pd.read_csv("data/framework_alignment_gs_gpt_4o.csv", encoding="utf-8")

# Check columns
if not {"answer_framework1", "gold_standard_answer"}.issubset(df_final.columns):
    raise ValueError("Expected columns 'answer_framework1' and 'gold_standard_answer' not found.")

# 2. Prepare Lists of Predictions and References
predictions = df_final["answer_framework1"].fillna("").tolist()       # The "system" or "framework" answers
references = df_final["gold_standard_answer"].fillna("").tolist()     # The gold-standard answers

# 3. Compute BERTScore
# model_type can be changed to a different model, e.g., "roberta-large", "microsoft/deberta-xlarge-mnli", etc.
# Check https://github.com/Tiiiger/bert_score for recommended models
P, R, F1 = score(predictions, references, model_type="microsoft/deberta-base-mnli")

# BERTScore returns a score for each prediction-reference pair. 
# P, R, F1 are torch tensors. You can store them row by row in your DataFrame.
df_final["bertscore_precision"] = [p.item() for p in P]
df_final["bertscore_recall"]    = [r.item() for r in R]
df_final["bertscore_f1"]        = [f.item() for f in F1]

# 4. Print or Store the Overall Averages
avg_p  = torch.mean(P).item()
avg_r  = torch.mean(R).item()
avg_f1 = torch.mean(F1).item()

print(f"BERTScore Average Precision: {avg_p:.4f}")
print(f"BERTScore Average Recall:    {avg_r:.4f}")
print(f"BERTScore Average F1:        {avg_f1:.4f}")

# 5. (Optional) Save the DataFrame with BERTScore columns
df_final.to_csv("data/framework_alignment_BERT_gs_gpt_4o.csv", index=False, encoding="utf-8")

print("Done! The DataFrame now has BERTScore columns for each pair of answers.")
