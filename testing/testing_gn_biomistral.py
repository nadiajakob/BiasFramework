import os
import requests
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Ensure these environment variables are set or hardcode them
HF_API_KEY = os.getenv("HF_API_KEY")
HF_BASE_URL = "https://d4ozqtnfgb5whvgv.us-east-1.aws.endpoints.huggingface.cloud/v1/"

# Initialize the OpenAI client for Hugging Face Endpoint
client = OpenAI(
    base_url=HF_BASE_URL,  
    api_key=HF_API_KEY
)

# Load your DataFrame containing the questions
gender_specific_questions = pd.read_csv('data/gender_neutral_questions.csv', encoding='utf-8')

# Initialize tqdm for pandas progress bar
tqdm.pandas()

def get_hf_response(question):
    """
    Query the Hugging Face endpoint using the OpenAI-like client for a given question.
    """
    # Construct messages as required by the API
    messages = [
    {"role": "user", "content": "You are a helpful AI assistant specializing in HIV-related questions. Provide a clear, accurate, and sensitive answer in 3 sentences.\n\n" + question}
    ]

    # Create a chat completion
    chat_completion = client.chat.completions.create(
        model="tgi",  # The provided model name from the example (may vary if you have a different model)
        messages=messages,
        max_tokens=250,
        temperature=0.2,
        top_p=0.9,
        stream=True,  # If you want streaming. Set to False if you prefer a single response.
        stop=None
    )

    # If streaming is enabled, consume the streamed response
    answer_chunks = []
    for message in chat_completion:
        part = message.choices[0].delta.content if message.choices[0].delta else ''
        if part:
            answer_chunks.append(part)

    final_answer = "".join(answer_chunks).strip()
    return final_answer

# Apply the function to 'question' column and create 'answer_llm' column
gender_specific_questions['answer_llm'] = gender_specific_questions['question'].progress_apply(get_hf_response)

# Save the updated DataFrame to a CSV file
gender_specific_questions.to_csv('data/gender_neutral_questions_biomistral.csv', index=False, encoding='utf-8')

print("LLM answers have been generated and saved in 'answer_llm' column.")
print("The updated CSV has been saved to 'data/gender_neutral_questions_biomistral.csv'.")

