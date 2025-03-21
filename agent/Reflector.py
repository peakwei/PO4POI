import logging
import json
import os
from openai import OpenAI,AsyncOpenAI,OpenAIError
from typing import List, Dict, Tuple, Optional, Any
import asyncio
import random
import re
import time
from tqdm import tqdm


class ReflectorAgent:
    def __init__(self, llm: str, seed: int, dataset_name: str, logger,
                 key: str, base: str, keep_reflection: bool = True):
        self.keep_reflection = keep_reflection
        self.memory = []
        self.attempt = 0
        self.dataset_name = dataset_name
        self.key = key
        self.base = base
        self.logger = logger
        self.seed = seed
        self.model_name=self.get_llm_model(llm)

    def get_llm_model(self,llm) -> str:
        llm_model_mapping = {
            'gpt': 'gpt-3.5-turbo',
            'gemini': 'gemini-pro',
            'moonshot': 'moonshot-v1-8k',
            'qwen': 'qwen-turbo',
            'claude': 'claude-3-5-sonnet-20240620',
            'llama': 'llama-2-70b'
        }
        return llm_model_mapping.get(llm, 'gpt-3.5-turbo')

    def create_client(self):
        client=OpenAI(
            api_key=self.key,
            base_url=self.base
        )
        return client

    def call_llm_api(self,client,user,system=None):
        if system:
            message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            content =  user
            message = [{"role": "user", "content": content}]
        for delay_secs in (2 ** x for x in range(0, 10)):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    temperature=0.2,
                    frequency_penalty=0.0)
                break
            except OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue
        return response.choices[0].message.content


    def get_refined_prompts(self,Initial_prompt,err_list,reasons):
        candidate_prompts=[]
        client=self.create_client()
        for err in tqdm(err_list):
            tmp_prompt="I'm trying to write a zero-shot recommender(predictor) prompt.\n" \
                        "My current prompt is \"$prompt$\"\n" \
                        "But this prompt cannot effectively capture the user's behavior patterns(Not in Top 10 recommended places).Here is user's long-term and short-term check-in records and candidate set: $error_case$ \n" \
                        "Here is the predicted top 10 places:$recommendation$.\n"\
                        "And Here is the target place that user will visit next:$Place$.\n"\
                        "give $num_feedbacks$ reasons why the prompt could have gotten this example wrong.\n" \
                        "Wrap each reason with <START> and <END>"
            tmp_prompt=tmp_prompt.replace('$prompt$',Initial_prompt)
            content = tmp_prompt.replace("$error_case$", err[0]).replace("$recommendation$",json.dumps(err[1]))
            content=content.replace("$Place$", str((err[2][0],err[2][2]))).replace('$num_feedbacks$',str(reasons))
            gradient=self.call_llm_api(client,content)
            tmp_prompt="I'm trying to write a zero-shot recommender prompt.\n" \
                       "My current prompt is $prompt$\n" \
                       "But this prompt cannot effectively capture the user's behavior patterns: $error_case$\n" \
                       "Based on these example the problem with this prompt is that $reasons$.\n" \
                       "Based on the above information, please wrote one improved prompt.\n"\
                       "Please consider not only the current user but also take into account the behavior and preferences of all other users when generating the prompt\n"\
                       "The prompt is wrapped with <START> and <END>.\n" \
                       "The new prompt is:"
            tmp_prompt=tmp_prompt.replace('$prompt$',Initial_prompt)
            tmp_prompt = tmp_prompt.replace("$error_case$", err[0])
            content = tmp_prompt.replace("$reasons$", gradient)
            edit_prompt=self.call_llm_api(client,content)
            edit_prompt_list=self.extract_edit_prompt(edit_prompt)
            candidate_prompts.extend(edit_prompt_list)
        try:
            sample_candidate_prompts=random.sample(candidate_prompts,16)
        except:
            sample_candidate_prompts=candidate_prompts
        return sample_candidate_prompts


    def extract_edit_prompt(self,response):
        pattern = r'<START>\s*(.*?)\s*<END>'
        result_list = re.findall(pattern, response, re.DOTALL)
        if len(result_list) == 0:
            pattern = r'<START>(.*?)<END>'
            result_list = re.findall(pattern, response, re.DOTALL)
        return result_list