import random
import math
import numpy as np
from utils import create_async_client
from openai import OpenAI,OpenAIError
from agent.Client import Client
class EvaluatorAgent:
    def __init__(self, llm: str, logger,
                 key: str, base: str):
        self.key = key
        self.base = base
        self.logger = logger
        self.llm=llm
        self.client=Client(self.key,self.base,self.llm)


    def ndcg(self,target_index):
        res = 1 / np.log2(target_index + 1)
        return res


    async def calculate_reward(self,prompt,sample_data):
        results=await self.client.process_trajectories(sample_data,prompt)
        reward=0
        for result in results:
            groundTruth = result[2][0]
            if groundTruth in result[1]:
                index = result[1].index(groundTruth) + 1
            else:
                index=-1
            reward = reward + self.ndcg(index)
        print(reward)
        return reward



    async def UCB_evaluation(self,sample_candidate_prompts,eval_data,top_b=8):
        numbers_of_selections =[0]*len(sample_candidate_prompts)
        sums_of_reward=[0]*len(sample_candidate_prompts)
        index_list = [i for i in range(len(sample_candidate_prompts))]
        for t in range(1, 33):
            sample_data = random.sample(eval_data, 256)
            if t == 1:
                select_prompt_index = random.choice(index_list)
            else:
                explore_param = 2
                results = [q_value + explore_param * math.sqrt(math.log(t) / (n + 1)) for q_value, n in
                           zip(sums_of_reward, numbers_of_selections)]
                max_result = max(results)
                select_prompt_index = results.index(max_result)
            select_prompt = sample_candidate_prompts[select_prompt_index]
            select_prompt_reward = await self.calculate_reward(select_prompt, sample_data)
            numbers_of_selections[select_prompt_index] += 64
            sums_of_reward[select_prompt_index] += select_prompt_reward / numbers_of_selections[select_prompt_index]

        if top_b>len(sample_candidate_prompts):
            raise Exception("The value of beamwidth needs to be less than the length of the prompt list")
        else:
            pairs = list(zip(sums_of_reward, sample_candidate_prompts))
            pairs.sort(reverse=True)
            top_b_prompt = [pair[1] for pair in pairs[:top_b]]

        return top_b_prompt


    async def generate_argmax_prompt(self,beam_candidate,eval_data):
            reward_list = [0] * len(beam_candidate)
            for index, prompt in enumerate(beam_candidate):
                reward = await self.calculate_reward(prompt, eval_data)
                if reward > len(eval_data) or reward < 0:
                    reward = 0
                reward_list[index] = reward
            prompt_index = reward_list.index(max(reward_list))

            return beam_candidate[prompt_index]






