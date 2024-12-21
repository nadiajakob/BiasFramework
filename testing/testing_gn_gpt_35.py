import os
import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

gender_specific_questions = pd.read_csv('data/gender_neutral_questions.csv', encoding='utf-8')

# Initialize tqdm for pandas progress bar
tqdm.pandas()

#--------------------------------------------#
# Answer LLM
#--------------------------------------------#

def get_llm_response(user_message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant specialized in answering HIV-related questions. Your task is to provide clear, accurate, and sensitive answers based on the user's question. Keep your answer short within 5 sentences "},
            {"role": "user", "content": user_message}
        ],
        max_tokens=100,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# This adds a new column 'answer_llm' with the model's answer for each question
gender_specific_questions['answer_llm'] = gender_specific_questions['question'].progress_apply(get_llm_response)

gender_specific_questions.to_csv('data/gender_neutral_questions_gpt_35.csv', index=False, encoding='utf-8')

print("LLM answers have been generated and saved in 'answer_llm' column for the gender_specific_questions DataFrame.")
print("The updated CSV has been saved to 'data/gender_neutral_questions_gpt_35.csv'.")

#-------------------------------------------#
# Framework
#-------------------------------------------#

API_URL = "http://localhost:3000/api/v1/prediction/c0bb58d3-2726-4cc7-8ceb-03d3e5cf14ed"

def query(payload):
    response = requests.post(API_URL, json=payload) 
    response.raise_for_status()  
    return response.json()

# add a new column to store details from the Framework
answerFramework = []
chatIdsFramework = []
sessionIdsFramework = []
agentReasoningsFramework = []

# Loop over each question in the DataFrame
for idx, row in tqdm(gender_specific_questions.iterrows(), total=gender_specific_questions.shape[0]):
    question = row['question']
    answer = row['answer_llm']  # Ensure that your DataFrame has this column from previous steps

    # Create a combined prompt that includes both the question and the answer
    combined_prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
    )

    try:
        # Send the combined prompt to the API
        output = query({"question": combined_prompt})

        # Extract desired fields from the response
        answerFramework.append(output.get('text', ''))
        chatIdsFramework.append(output.get('chatId', ''))
        sessionIdsFramework.append(output.get('sessionId', ''))
        agentReasoningsFramework.append(output.get('agentReasoning', ''))
    except Exception as e:
        print(f"Error querying question at index {idx}: {e}")
        # Append empty strings in case of error to keep lists consistent
        answerFramework.append('')
        chatIdsFramework.append('')
        sessionIdsFramework.append('')
        agentReasoningsFramework.append('')

# Add the responses back into the DataFrame
gender_specific_questions['answer_framework1'] = answerFramework
gender_specific_questions['chatId_framework1'] = chatIdsFramework
gender_specific_questions['sessionId_framework1'] = sessionIdsFramework
gender_specific_questions['agentReasonings_framework1'] = agentReasoningsFramework

# Save the updated DataFrame
gender_specific_questions.to_csv('data/gender_neutral_questions_gpt_35.csv', index=False, encoding='utf-8')

print(gender_specific_questions)