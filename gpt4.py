import google.generativeai as genai
import time
import logging
import json
import os

class GPT4:

    def __init__(self, max_tokens=1024, temperature=0.0, logprobs=None, n=1, engine='gemini-2.0-flash-exp',
        frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False, **kwargs):

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rstrip = rstrip
        self.engine = engine
        
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY', 'your_google_api_key_here')
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.engine,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json"
            )
        )

    def complete(self, prompt):

        if self.rstrip:
            prompt = prompt.rstrip()
        retry_interval_exp = 1

        system_prompt = "You are an expert in control engineering design. Respond only with valid JSON."
        full_prompt = f"{system_prompt}\n\n{prompt}"

        while True:
            try:
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                if "rate" in str(e).lower() or "quota" in str(e).lower():
                    logging.warning(f"Rate limit or quota error: {e}. Retrying...")
                    time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                    retry_interval_exp += 1
                elif "connection" in str(e).lower() or "network" in str(e).lower():
                    logging.warning(f"Connection error: {e}. Retrying...")
                    time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                    retry_interval_exp += 1
                else:
                    logging.error(f"Unexpected error: {e}")
                    raise e
