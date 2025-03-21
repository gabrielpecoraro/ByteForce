import openai

def initialize_llm(api_key):
    openai.api_key = api_key

def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']