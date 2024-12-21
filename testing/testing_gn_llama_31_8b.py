# import os
# import requests
# import pandas as pd
# from groq import Groq
# from tqdm import tqdm
# import time
# from dotenv import load_dotenv

# load_dotenv() 

# # # Initialize the Groq client
# # client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# # # Load your DataFrame containing the questions
# # gender_specific_questions = pd.read_csv('data/gender_neutral_questions.csv', encoding='utf-8')

# # # Initialize tqdm for pandas progress bar
# # tqdm.pandas()

# # # Set frequency and duration for pauses
# # PAUSE_FREQUENCY = 5
# # PAUSE_DURATION = 60

# # # Global request counter
# # request_count = 0

# # def get_groq_response(question):
# #     """
# #     Query the Groq model for a given question and return the response.
# #     """
# #     messages = [
# #         {"role": "system", "content": "You are a helpful AI assistant specialized in answering HIV-related questions. Provide a clear, accurate, and sensitive answer in 3 sentences."},
# #         {"role": "user", "content": question}
# #     ]

# #     completion = client.chat.completions.create(
# #         model="llama-3.1-8b-instant",    
# #         messages=messages,
# #         temperature=0.2,
# #         max_tokens=250,
# #         top_p=1,
# #         stream=True,
# #         stop=None
# #     )

# #     answer = []
# #     for chunk in completion:
# #         part = chunk.choices[0].delta.content if chunk.choices[0].delta else ''
# #         if part:
# #             answer.append(part)

# #     final_answer = "".join(answer).strip()
# #     return final_answer

# # def process_question(q):
# #     global request_count
# #     request_count += 1
# #     if request_count % PAUSE_FREQUENCY == 0:
# #         print(f"Reached {request_count} requests. Pausing for {PAUSE_DURATION} seconds to avoid rate limits...")
# #         time.sleep(PAUSE_DURATION)
# #     return get_groq_response(q)

# # # Apply the function to 'question' column and create 'answer_llm' column
# # gender_specific_questions['answer_llm'] = gender_specific_questions['question'].progress_apply(process_question)

# # # Save the updated DataFrame to a CSV file
# # gender_specific_questions.to_csv('data/gender_neutral_questions_llama_31_8b.csv', index=False, encoding='utf-8')

# # print("LLM answers have been generated and saved in 'answer_llm' column.")
# # print("The updated CSV has been saved to 'data/gender_neutral_questions_llama_31_8b.csv'.")


# # #-------------------------------------------#
# # # Framework  
# # #-------------------------------------------#
# gender_specific_questions = pd.read_csv('data/gender_neutral_questions_llama_31_8b.csv', encoding='utf-8')
# tqdm.pandas()


# API_URL = "http://localhost:3000/api/v1/prediction/c0bb58d3-2726-4cc7-8ceb-03d3e5cf14ed"

# def query(payload):
#     response = requests.post(API_URL, json=payload) 
#     response.raise_for_status()  
#     return response.json()

# # add a new column 'answer_framework' 
# answerFramework = []
# chatIdsFramework = []
# sessionIdsFramework = []
# agentReasoningsFramework = []

# request_count = 0  # Global request counter

# # Loop over each question in the DataFrame
# for idx, row in tqdm(gender_specific_questions.iterrows(), total=gender_specific_questions.shape[0]):
#     question = row['question']

#     # Increment request count and check if we need to pause
#     request_count += 1
#     if request_count % 2 == 0:
#         print(f"Reached {request_count} requests. Pausing for 60 seconds to avoid rate limits...")
#         time.sleep(60)

#     try:
#         # Send the combined prompt to the API
#         output = query({"question": question})

#         # Extract desired fields
#         answerFramework.append(output.get('text', ''))
#         chatIdsFramework.append(output.get('chatId', ''))
#         sessionIdsFramework.append(output.get('sessionId', ''))
#         agentReasoningsFramework.append(output.get('agentReasoning', ''))
#     except Exception as e:
#         print(f"Error querying question at index {idx}: {e}")
#         # Append empty strings in case of error to maintain consistent lengths
#         answerFramework.append('')
#         chatIdsFramework.append('')
#         sessionIdsFramework.append('')
#         agentReasoningsFramework.append('')

# # Add the responses back into the DataFrame
# gender_specific_questions['answer_framework1'] = answerFramework
# gender_specific_questions['chatId_framework1'] = chatIdsFramework
# gender_specific_questions['sessionId_framework1'] = sessionIdsFramework
# gender_specific_questions['agentReasonings_framework1'] = agentReasoningsFramework

# # Save the updated DataFrame
# gender_specific_questions.to_csv('data/gender_neutral_questions_llama_31_8b.csv', index=False, encoding='utf-8')

# print(gender_specific_questions)

import os
import requests
import pandas as pd
from tqdm import tqdm
import time

# Load the DataFrame that already has some answers
gender_specific_questions = pd.read_csv('data/gender_neutral_questions_llama_31_8b.csv', encoding='utf-8')

# Filter rows where 'answer_framework1' is NaN or empty
mask_unanswered = gender_specific_questions['answer_framework1'].isna() | (gender_specific_questions['answer_framework1'].str.strip() == '')
unanswered_df = gender_specific_questions[mask_unanswered].copy()

print(f"Found {len(unanswered_df)} unanswered questions. Proceeding to query the API for these.")

API_URL = "http://localhost:3000/api/v1/prediction/c0bb58d3-2726-4cc7-8ceb-03d3e5cf14ed"

def query(payload):
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()

answerFramework = []
chatIdsFramework = []
sessionIdsFramework = []
agentReasoningsFramework = []

# Rate limiting setup: Pause after every 3 requests for 60 seconds
PAUSE_FREQUENCY = 2
PAUSE_DURATION = 60
request_count = 0

for idx, row in tqdm(unanswered_df.iterrows(), total=unanswered_df.shape[0]):
    question = row['question']
    answer = row['answer_llm']  # Ensure this column is present from previous steps

    combined_prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
    )

    # Rate limit check
    request_count += 1
    if request_count % PAUSE_FREQUENCY == 0:
        print(f"Reached {request_count} requests. Pausing for {PAUSE_DURATION} seconds...")
        time.sleep(PAUSE_DURATION)

    try:
        output = query({"question": combined_prompt})
        answerFramework.append(output.get('text', ''))
        chatIdsFramework.append(output.get('chatId', ''))
        sessionIdsFramework.append(output.get('sessionId', ''))
        agentReasoningsFramework.append(output.get('agentReasoning', ''))
    except Exception as e:
        print(f"Error querying question at index {idx}: {e}")
        answerFramework.append('')
        chatIdsFramework.append('')
        sessionIdsFramework.append('')
        agentReasoningsFramework.append('')

# Now we have new responses for the previously unanswered questions
unanswered_df['answer_framework1'] = answerFramework
unanswered_df['chatId_framework1'] = chatIdsFramework
unanswered_df['sessionId_framework1'] = sessionIdsFramework
unanswered_df['agentReasonings_framework1'] = agentReasoningsFramework

# Merge the updated answers back into the original DataFrame
# We'll do this by index. The unanswered_df should have the same index as in the original DF
for col in ['answer_framework1', 'chatId_framework1', 'sessionId_framework1', 'agentReasonings_framework1']:
    gender_specific_questions.loc[unanswered_df.index, col] = unanswered_df[col]

# Save the fully updated DataFrame
gender_specific_questions.to_csv('data/gender_neutral_questions_llama_31_8b.csv', index=False, encoding='utf-8')

print("Data processing complete. Updated DataFrame saved with newly answered questions.")