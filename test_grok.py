from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1")
prompt = "Output ONLY: {{\"test\": \"success\"}}"
response = client.chat.completions.create(model="grok-3-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0)
print(response.choices[0].message.content)