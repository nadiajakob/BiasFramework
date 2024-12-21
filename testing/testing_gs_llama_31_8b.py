import os
import requests
import time
import pandas as pd
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv() 

# Initialize the Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))


# Load your DataFrame containing the questions
gender_specific_questions = pd.read_csv('data/gender_specific_questions.csv', encoding='utf-8')

# Initialize tqdm for pandas progress bar
tqdm.pandas()

#---------------------------------------#
# Answer LLM llama 3.1 8b instant
#---------------------------------------#

def get_groq_response(question):
    """
    Query the Groq model for a given question and return the response.
    """

    # Construct your system/user messages as needed. Here's a simple prompt:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in answering HIV-related questions. Provide a clear, accurate, and sensitive answer in 3 sentences."},
        {"role": "user", "content": question}
    ]

    # Create a chat completion
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",    
        messages=messages,
        temperature=0.2,          # Adjust parameters as desired
        max_tokens=250,
        top_p=1,
        stream=True,              # The example shows streaming
        stop=None
    )

    # Consume the streamed response
    answer = []
    for chunk in completion:
        # Each chunk contains partial response
        # chunk.choices[0].delta.content might be None or partial string
        part = chunk.choices[0].delta.content
        if part:
            answer.append(part)

    # Join all parts of the streamed answer
    final_answer = "".join(answer).strip()
    return final_answer

# Apply the function to 'question' column and create 'answer_llm' column
gender_specific_questions['answer_llm'] = gender_specific_questions['question'].progress_apply(get_groq_response)

# Save the updated DataFrame to a CSV file
gender_specific_questions.to_csv('data/gender_specific_questions_llama_31_8b.csv', index=False, encoding='utf-8')

print("LLM answers have been generated and saved in 'answer_llm' column.")
print("The updated CSV has been saved to 'data/gender_specific_questions_llama_31_8b.csv'.")

# #-------------------------------------------#
# # Framework  
# #-------------------------------------------#

gender_specific_questions = pd.read_csv('data/gender_specific_questions_llama_31_8b.csv', encoding='utf-8')
tqdm.pandas()


API_URL = "http://localhost:3000/api/v1/prediction/c0bb58d3-2726-4cc7-8ceb-03d3e5cf14ed"

def query(payload):
    response = requests.post(API_URL, json=payload) 
    response.raise_for_status()  
    return response.json()

# add a new column 'answer_framework' 
answerFramework = []
chatIdsFramework = []
sessionIdsFramework = []
agentReasoningsFramework = []

request_count = 0  # Global request counter

# Loop over each question in the DataFrame
for idx, row in tqdm(gender_specific_questions.iterrows(), total=gender_specific_questions.shape[0]):
    question = row['question']

    # Increment request count and check if we need to pause
    request_count += 1
    if request_count % 3 == 0:
        print(f"Reached {request_count} requests. Pausing for 60 seconds to avoid rate limits...")
        time.sleep(60)

    try:
        # Send the combined prompt to the API
        output = query({"question": question})
         # Extract desired fields
        answerFramework.append(output.get('text', ''))
        chatIdsFramework.append(output.get('chatId', ''))
        sessionIdsFramework.append(output.get('sessionId', ''))
        agentReasoningsFramework.append(output.get('agentReasoning', ''))
    except Exception as e:
        print(f"Error querying question at index {idx}: {e}")
        # Append empty strings in case of error to maintain consistent lengths
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
gender_specific_questions.to_csv('data/gender_specific_questions_llama_31_8b.csv', index=False, encoding='utf-8')

print(gender_specific_questions)