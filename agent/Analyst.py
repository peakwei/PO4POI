from typing import List, Dict, Tuple, Optional, Any
from utils import haversine_distance,create_async_client
import logging
from openai import OpenAI,AsyncOpenAI, OpenAIError
import json
import os
import asyncio
import random
from agent.Client import Client
class Analyst:
    def __init__(self, llm: str, logger: logging.Logger, temperature: float, key: str, base: str) -> None:
        self.logger = logger
        self.llm=llm
        self.temperature = temperature
        self.key = key
        self.base = base
        self.client=Client(self.key,self.base,self.llm)
        self.metric={'hit1':0,'hit5':0,'hit10':0,'rr':0}


    async def analyze_initial_batch(self,Initial_prompt,Initial_batch,t=5):

        results=await self.client.process_trajectories(Initial_batch,Initial_prompt)
        err_list=self.analyze_error(results,t)
        return err_list


    def analyze_error(self,results,t=5):
        err_list=[]
        for result in results:
            groundTruth =result[2][0]
            if groundTruth in result[1]:
                index = result[1].index(groundTruth)+1
                if index > t:
                    err_list.append(result)
            else:
                err_list.append(result)
        return err_list

    async def analyze_final_result(self,top_prompt,test_data):
        err=[]
        results=await self.client.process_trajectories(test_data,top_prompt)
        for result in results:
            groundTruth = result[2][0]
            if groundTruth in result[1]:
                index = result[1].index(groundTruth) + 1
                if index==1:
                    self.metric['hit1']+=1
                if index<=5:
                    self.metric['hit5']+=1
                self.metric['hit10']+=1
                self.metric['rr']+=1/index
            else:
                err.append(result)

        return self.metric, err







