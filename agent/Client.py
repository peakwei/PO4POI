from openai import OpenAI,OpenAIError,AsyncOpenAI
import asyncio
import random
import re
class Client:
    def __init__(self,key,base,llm):
        self.key=key
        self.base=base
        self.model_name=self.get_llm_model(llm)
        self.client=AsyncOpenAI(
            api_key=self.key,
            base_url=self.base
        )

    def get_llm_model(self, llm) -> str:
        llm_model_mapping = {
            'gpt': 'gpt-3.5-turbo',
            'gemini': 'gemini-pro',
            'moonshot': 'moonshot-v1-8k',
            'qwen': 'qwen-turbo',
            'claude': 'claude-3-5-sonnet-20240620',
            'llama': 'llama-2-70b'
        }
        return llm_model_mapping.get(llm, 'gpt-3.5-turbo')


    async def request_recommendation(self, user, system=None):
        if system:
            message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            content =  user
            message = [{"role": "user", "content": content}]
        for delay_secs in (2 ** x for x in range(0, 10)):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    temperature=0.0,
                    frequency_penalty=0.0
                )
                break
            except OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                await asyncio.sleep(sleep_dur)
                continue

        content = response.choices[0].message.content
        pattern = r'\{[^\{\}]*\}'
        responses = re.findall(pattern, content, re.DOTALL)
        try:
            res = eval(responses[0])
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
        return res['recommendation']
        ##try:
          ##  ## Ensure the content is valid and contains the "recommendation" key
            ##result = eval(content)
            ##if isinstance(result, dict) and "recommendation" in result:
              ##  return result["recommendation"]
            ##else:
              ##  print("No 'recommendation' key in LLM response.")
               ## return []
        ##except Exception as e:
          ##  print(f"Failed to parse LLM response: {e}")
            ##return []

    async def process_trajectory(self, trajectory, prompt):
        response = await self.request_recommendation(user=trajectory['input'], system=prompt)
        return (trajectory['input'], response, trajectory['target'])

    async def preprocess_batch(self,batch, prompt):
        tasks = [self.process_trajectory(trajectory, prompt) for trajectory in batch]
        results = await asyncio.gather(*tasks)
        return results

    async def process_trajectories(self, data, prompt, batch_size=100):
        all_results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]  # each time process one batch
            print(f"Processing batch {i // batch_size + 1}...")
            results = await self.preprocess_batch( batch, prompt)
            all_results.extend(results)
        return all_results