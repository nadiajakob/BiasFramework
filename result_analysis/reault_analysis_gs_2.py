import pandas as pd

# 1. Load each dataset into a separate DataFrame
df_3_5 = pd.read_csv('data\evaluated_responses_gs_3.5_2.csv', encoding='utf-8')
df_4o = pd.read_csv('data/evaluated_responses_gs_4o.csv', encoding='utf-8')
df_llama_8b = pd.read_csv('data/evaluated_responses_gs_llama_31_8b.csv', encoding='utf-8')

# 2. Add a 'model' column to each
df_3_5['model'] = 'gpt-3.5-turbo'
df_4o['model'] = 'gpt-4o'
df_llama_8b['model'] = 'llama-3.1-8b'

# 3. Concatenate all DataFrames
df_all = pd.concat([df_3_5, df_4o, df_llama_8b], ignore_index=True)

# 4. Group by 'model' and 'source' to compute average bias score
#    Note: Some rows may have missing bias_score, so skipna by default is True for mean()
df_grouped = df_all.groupby(['model', 'source'], dropna=True)['bias_score'].mean().reset_index()

# 5. Pivot the result so that rows are 'model' and columns are 'source'
pivot_table = df_grouped.pivot(index='model', columns='source', values='bias_score')

# 6. Print the pivot table
print("Average Bias Score by Model and Source:")
print(pivot_table)
