import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class OpenAIModel:
    def __init__(self, model_id, client=None):
        self.model = model_id
        if not client:
            self.client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"], organization=os.environ.get("OPENAI_ORG", None)
            )
        else:
            self.client = client

    def request_llm(self, prompt, system_prompt: object = None, response_format: object = None, max_tokens:int =4096):
        messages = []
        if system_prompt:
            messages.append({"role": "system",
                            "content": [{
                                "type": "text",
                                "text": system_prompt
                            }]})

        messages.append({"role": "user", "content": prompt})

        if response_format:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0,
                max_tokens=max_tokens,
                messages=messages,
                response_format=response_format
            )
            message = response.choices[0].message.content
            return message

        else:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0,
                max_tokens=max_tokens,
                messages=messages
            )

            text = response.choices[0].message.content
            return text