import json
from agent.Evaluator import EvaluatorAgent
from agent.DataAgent import DataAgent
from agent.Analyst import Analyst
from agent.Manager import ManageAgent
import pandas as pd
from logging import getLogger
import asyncio
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger=getLogger()
    da = DataAgent('nyc', case_num=10000)
    an = Analyst('gpt', logger, 0, key=, base=)
    ma = ManageAgent(da, an, 42, 'gpt', 'nyc', key=,base=)
    ma.split_data()
    asyncio.run(ma.run_poi_workflow())






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
