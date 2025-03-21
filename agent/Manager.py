from typing import List, Dict, Union, Any, Tuple, Optional
from agent.Reflector import ReflectorAgent
from agent.Evaluator import EvaluatorAgent
from tqdm import tqdm
from logging import getLogger
import random
import asyncio
from utils import haversine_distance
import json
class ManageAgent:
    def __init__(self, data_agent, analyst, evaluator,seed: int,
                 llm: str, datasetName: str,
                 key: str, base: str):

        self.data_agent = data_agent
        self.analyst = analyst
        self.evaluator= evaluator
        self.llm = None

        # Data Agent
        self.datasetName = datasetName


        # logger
        self.seed = seed
        self.logger=getLogger()
        self.logger.propagate = False

        # Initial
        self.llm = self.set_llm(llm)
        self.key = key
        self.base = base

    def set_llm(self, llm_name: str) -> str:
            llm_models = {
                'gpt': 'GPT model',
                'gemini': 'Gemini model',
                'moonshot': 'MoonShot model',
                'qwen': 'Qwen model',
                'claude': 'Claude model'
            }

            if llm_name in llm_models:
                self.logger.info(f"Manager starting {llm_models[llm_name]}...")
                return llm_name
            else:
                self.logger.error(f"Unknown LLM model: {llm_name}. Defaulting to GPT model.")
                return 'gpt'

    def split_data(self):
        trajectories=self.data_agent.construct_trajectories()
        random.seed(self.seed)
        Initial_batch = random.sample(trajectories, 100)
        with open(f'./dataset/{self.datasetName}/Initial_batch.json', 'w') as f:
            json.dump(Initial_batch, f)
        rest = [trajectory for trajectory in trajectories if trajectory not in Initial_batch]
        random.seed(self.seed)
        eval_data=random.sample(rest,500)
        with open(f'./dataset/{self.datasetName}/eval_data.json', 'w') as f:
            json.dump(eval_data, f)
        test_data = [sample for sample in rest if sample not in eval_data]
        with open(f'./dataset/{self.datasetName}/test_data.json', 'w') as f:
            json.dump(test_data, f)
        return

    async def run_poi_workflow(self,t=5):
        with open(f'./dataset/{self.datasetName}/Initial_batch.json','r') as f:
            Initial_batch=json.load(f)
        with open(f'./dataset/{self.datasetName}/eval_data.json','r') as f:
            eval_data=json.load(f)
        with open(f'./dataset/{self.datasetName}/test_data.json', 'r') as f:
            test_data = json.load(f)
        Initial_prompt = f"""
                Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
        The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
        Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

        Requirements:
        1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
        2. Consider the recent check-ins to extract users' current perferences.
        3. Consider the "Distance" since people tend to visit nearby pois.
        4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.
        
        Please organize your answer in a JSON object containing following keys:
        "recommendation" (10 distinct POIIDs of the ten high probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
        """
        err_list=await self.analyst.analyze_initial_batch(Initial_prompt,Initial_batch,t)
        reflector_agent = ReflectorAgent(llm=self.llm, logger=self.logger,
                                         dataset_name=self.datasetName, seed=self.seed
                                         , key=self.key, base=self.base)
        sample_candidate_prompts=reflector_agent.get_refined_prompts(Initial_prompt,err_list,3)
        evaluator_agent=EvaluatorAgent(key=self.key,base=self.base,llm=self.llm,logger=self.logger)
        top_b_prompts=await evaluator_agent.UCB_evaluation(sample_candidate_prompts,eval_data)
        top_prompt=await evaluator_agent.generate_argmax_prompt(top_b_prompts,eval_data)
        with open('dataset/nyc/final_prompt.json','w') as f:
            json.dump(top_prompt,f,ensure_ascii=False)
        metric,err=await self.analyst.analyze_final_result(top_prompt,test_data)
        num_trajectories=len(test_data)
        print(f"acc@1: {metric['hit1']/num_trajectories}, acc@5: {metric['hit5']/num_trajectories}, acc@10: {metric['hit10']/num_trajectories}, mrr@10: {metric['rr']/num_trajectories}")
        return metric,err

    async def run_baseline(self):
        with open(f'./dataset/{self.datasetName}/test_data.json', 'r') as f:
            test_data = json.load(f)
        print(len(test_data))
        Initial_prompt = f"""
                       Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
               The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
               Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

               Requirements:
               1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
               2. Consider the recent check-ins to extract users' current perferences.
               3. Consider the "Distance" since people tend to visit nearby pois.
               4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.

               Please organize your answer in a JSON object containing following keys:
               "recommendation" (10 distinct POIIDs of the ten high probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
               """
        metric,err=await self.analyst.analyze_final_result(Initial_prompt,test_data)
        num_trajectories = len(test_data)
        print(f"acc@1: {metric['hit1'] / num_trajectories}, acc@5: {metric['hit5'] / num_trajectories}, acc@10: {metric['hit10'] / num_trajectories}, mrr@10: {metric['rr'] / num_trajectories}")

