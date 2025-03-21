import os
import random
from typing import Tuple, Optional, Dict, List, Any
from utils import haversine_distance
from tqdm import tqdm
import json

class DataAgent:

    def __init__(self,dataset_name:str, case_num: int, k: int =100, traj_thre: int =5, filepath: Optional[str] = None) -> None:
        self.dataset_name=dataset_name
        self.case_num=case_num
        self.k=k
        self.traj_thre=traj_thre
        self.filepath=filepath
        self.train_path, self.test_path= self.getPaths()


    def getPaths(self) -> Tuple[str, str]:
        if self.dataset_name == 'nyc':
            self.filepath = './dataset/nyc/nyc_{}_sample.csv'
        elif self.dataset_name == 'tky':
            self.filepath = './dataset/tky/tky_{}_sample.csv'
        else:
            raise NotImplementedError(f'Dataset {self.dataset_name} is not implemented')

        trainpath = os.path.join(os.path.dirname(self.filepath), f'{self.dataset_name}_train_sample.csv')
        testpath = os.path.join(os.path.dirname(self.filepath), f'{self.dataset_name}_test_sample.csv')

        return trainpath, testpath

    def readTrain(self) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Dict[str, str]]]:

        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training data file not found: {self.train_path}")
        print(f"Trying to open training data file: {self.train_path}")
        longs = dict()
        pois = dict()
        with open(self.train_path, 'r') as file:
                lines = file.readlines()
        for line in lines[1:]:
                data = line.strip().split(',')
                time, u, lati, longi, i, category = data[1], data[5], data[6], data[7], data[8], data[10]
                if i not in pois:
                    pois[i] = {"latitude": lati, "longitude": longi, "category": category}
                if u not in longs:
                    longs[u] = list()
                longs[u].append((i, time))
        return longs, pois

    def readTest(self) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Dict[str, str]]]:

        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data file not found: {self.test_path}")
        print(f"Trying to open test data file: {self.test_path}")
        recents = dict()
        pois = dict()
        targets = dict()
        traj2u = dict()
        with open(self.test_path, 'r') as file:
            lines = file.readlines()
        for line in lines[1:]:
            data = line.strip().split(',')
            time, trajectory, u, lati, longi, i, category = data[1], data[3], data[5], data[6], data[7], data[8], data[
                10]
            if i not in pois:
                pois[i] = {"latitude": lati, "longitude": longi, "category": category}
            if trajectory not in traj2u:
                traj2u[trajectory] = u
            if trajectory not in recents:
                ##recents[trajectory] = list()
                ##recents[trajectory].append((i, time))
                ##targets[trajectory] = (i, time)
                recents[trajectory] = list()
                recents[trajectory].append((i, time))
            else:
                if trajectory in targets:
                    recents[trajectory].append(targets[trajectory])
                targets[trajectory] = (i, time)
        return recents, pois, targets, traj2u

    def getData(self) -> Dict[str, Any]:
        longs, poiInfos = self.readTrain()
        kb=self.construct_knowledge_base()
        recents, testpoi, targets, traj2u = self.readTest()
        poiInfos.update(testpoi)
        targets = dict(list(targets.items())[:self.case_num])

        return {
            "longs": longs,
            "recents": recents,
            "targets": targets,
            "knowledge_base": kb,
            "poiInfos": poiInfos,
            "traj2u": traj2u,
            "poiList": list(poiInfos.keys())
        }

    def getcompletetraj(self, traj2u, longs, recents, targets):  ##to get each complete traj(containing user longterm)
        c_traj = dict()
        for traj_id, traj in targets.items():
            user = traj2u[traj_id]
            longterm = longs[user]
            shortterm = recents[traj_id]
            complete = longterm + shortterm
            c_traj[traj_id] = complete
        return c_traj


    def gettrajpois(self,c_traj):  ##to get pois in each complete traj
        traj_pois = dict()
        for traj_id, traj in c_traj.items():
            r_traj = traj[::-1]
            traj_pois[traj_id] = list()
            for c in r_traj:
                if c[0] not in traj_pois[traj_id]:
                    traj_pois[traj_id].append(c[0])
        return traj_pois


    def getnearbypois(self,his_pois,pois,poiList,k=100):
        mostrec = his_pois[0]
        his = [(poi, haversine_distance(pois[poi]["latitude"], pois[poi]["longitude"], pois[mostrec]["latitude"],
                                        pois[mostrec]["longitude"]), pois[poi]["category"]) for poi in his_pois]
        if len(his) >= k:
            his.sort(key=lambda x: x[1])
            return his
        allpoi = [(poi, haversine_distance(pois[poi]["latitude"], pois[poi]["longitude"], pois[mostrec]["latitude"],
                                           pois[mostrec]["longitude"]), pois[poi]["category"]) for poi in poiList]
        allpoi.sort(key=lambda x: x[1])
        for poi in allpoi:
            if len(his) >= k:
                break
            if poi in his:
                continue
            his.append(poi)
        his.sort(key=lambda x: x[1])
        return his


    def getcans(self,traj_pois, pois, targets, k, poiList):
        cand = dict()
        for id, target in tqdm(targets.items()):
            his = traj_pois[id]
            groundtruth = target[0]
            if len(his) >= k:
                his = his[:k]
                if groundtruth not in his:
                    his = his[:k - 1]
                    his.append(groundtruth)
            else:
                if groundtruth not in his:
                    his.append(groundtruth)
            cans = self.getnearbypois(his, pois, poiList, k)
            cand[id] = cans
        return cand


    def construct_trajectories(self):
        trajectories=[]
        data=self.getData()
        c_traj=self.getcompletetraj(data['traj2u'],data['longs'],data['recents'],data['targets'])
        traj_pois=self.gettrajpois(c_traj)
        cand=self.getcans(traj_pois,data['poiInfos'],data['targets'],self.k,data['poiList'])
        retreived=self.get_retreived_trajs(data['targets'],data['knowledge_base']['user_knowledge_base'],data['traj2u'],data['recents'])
        filtered_retreived=self.filter_similar_trajs_by_time(data['targets'],data['recents'],retreived)
        filtered_retreived_w_cat=self.coding_with_cat(filtered_retreived,data['poiInfos'])
        for traj, groundTruth in data['targets'].items():
            seed_value=eval(traj)
            random.seed(seed_value)
            u=data['traj2u'][traj]
            long=data['longs'][u]
            rec=data['recents'][traj]
            time=rec[-1][1]
            longterm=[(poi,data['poiInfos'][poi]['category']) for poi,_ in long][-40:]
            recent=[(poi,data['poiInfos'][poi]['category']) for poi,_ in rec][-5:]
            negsample=random.sample(data['poiList'],100)
            ##candidateSet=negsample+[groundTruth[0]]
            candidateSet=cand[traj]
            similar=filtered_retreived_w_cat[traj]
            prompt=self.create_prompt_llmmove(longterm,recent,candidateSet,similar)

            ##prompt=self.create_prompt_mas(longterm,recent,candidateSet,u,time)

            trajectories.append({'input':prompt, 'target':(groundTruth[0],groundTruth[1],data['poiInfos'][groundTruth[0]]['category'])})
        self.save_trajectories_data(trajectories)
        print(trajectories[0])
        return trajectories


    def create_prompt_llmmove(self,longterm,recent,candidateSet,similar):
        prompt = f"""\
            <long-term check-ins> [Format: (POIID, Category)]: {longterm}
            <recent check-ins> [Format: (POIID, Category)]: {recent}
            <candidate set> [Format: (POIID, Distance, Category)]: {candidateSet}
            """
        return prompt

    def create_prompt_mas(self,longterm,recent,candidateSet,u,time):
        prompt = f"""\
        Your task is to recommend a user's next point-of-interest (POI) from {candidateSet} based on his/her trajectory information.
        <question> The following is a trajectory of user {u}: {recent}. \
        There is also historical data: {longterm}. Given the data, at {time}, which POI id \
        will user {u} visit? Note that POI id is an identifier in the set of POIs. \
        Please organize your answer in a JSON object containing the following keys:
        - "recommendation": a list of 10 most likely distinct POIIDs from the candidate set, in descending order of probability.
        """
        return prompt
    def save_trajectories_data(self,trajectories):
        if self.dataset_name=='nyc':
            save_path='./dataset/nyc/nyc_trajectories.json'
            with open(save_path,'w') as f:
                json.dump(trajectories,f,ensure_ascii=False)
        elif self.dataset_name=='tky':
            save_path = './dataset/tky/tky_trajectories.json'
            with open(save_path, 'w') as f:
                json.dump(trajectories, f, ensure_ascii=False)
        else:
            raise NotImplementedError(f'Dataset {self.dataset_name} is not implemented.')

    def construct_knowledge_base(self):
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training data file not found: {self.train_path}")
        print(f"Trying to open training data file: {self.train_path}")
        traj_items=dict()
        long_items=dict()
        traj2u=dict()
        with open(self.train_path, 'r') as file:
                lines = file.readlines()
        for line in lines[1:]:
                data = line.strip().split(',')
                time, trajectory, u, lati, longi, i, category = data[1], data[3], data[5], data[6], data[7], data[8], data[10]
                if trajectory not in traj_items:
                    traj_items[trajectory]=[]
                traj_items[trajectory].append(i)
                if trajectory not in traj2u:
                    traj2u[trajectory]=u
        for t,items in traj_items.items():
            u=traj2u[t]
            if u not in long_items:
                long_items[u]=[]
            long_items[u].append(items)
        kb ={
            'user_knowledge_base':long_items,
            'traj_knowledge': traj_items
        }
        return kb

    def get_retreived_trajs(self,targets,user_knowledge_base,traj2u,recents):
        similar_traj=dict()
        for traj, groundTruth in tqdm(targets.items()):
            rec=recents[traj][-3:]
            rec_item=[r[0] for r in rec]
            user=traj2u[traj]
            self_similar_traj=[]
            other_similar_traj=[]
            for u,seq_list in user_knowledge_base.items():
                for seq in seq_list:
                    if self.is_ordered_subsequence(rec_item,seq):
                        if user==u:
                            self_similar_traj.append(seq)
                        else:
                            other_similar_traj.append(seq)
            similar_traj[traj]={'self':self_similar_traj,'other':other_similar_traj}
        return similar_traj

    def is_ordered_subsequence(self,query, sequence):
        q_idx = 0
        for item in sequence:
            if q_idx < len(query) and item == query[q_idx]:
                q_idx += 1
            if q_idx == len(query):
                return True
        return False

    def filter_similar_trajs_by_time(self,targets, recents, retreived):
        for traj,_ in tqdm(targets.items()):
            recent = recents[traj]
            rec = [r[0] for r in recent][-3:]
            similar_traj = retreived[traj]
            if len(similar_traj['self']) > self.traj_thre:
                top_k_self_similar_traj = self.find_best_k_segment(rec,similar_traj['self'])
                self_similar_traj=[t['source'] for t in top_k_self_similar_traj]
                retreived[traj]['self']=self_similar_traj
            if len(similar_traj['other']) > self.traj_thre:
                top_k_other_similar_traj =self.find_best_k_segment(rec,similar_traj['other'])
                other_similar_traj = [t['source'] for t in top_k_other_similar_traj]
                retreived[traj]['other'] = other_similar_traj
        return retreived





    def find_best_k_segment(self,rec,trajs):
        results=[]
        for traj in trajs:
            positions=[]
            i=0
            for r in rec:
                try:
                    idx=traj.index(r,i)
                    positions.append(idx)
                    i=idx+1
                except ValueError:
                    break
            if len(positions)==len(rec):
                start = positions[0]
                end = positions[-1]
                span =end-start
                segment=traj[start:end+1]
                results.append({
                    'segment':segment,
                    'source': traj,
                    'span': span
                })
        top_k=sorted(results, key=lambda x:x['span'])[:self.traj_thre]
        return top_k


    def coding_with_cat(self,filtered,poiInfos):
        for traj, similar in filtered.items():
            self_similar_revised=[]
            self_similar=similar['self']
            other_similar_revised=[]
            other_similar=similar['other']
            for t in self_similar:
                r=[(p,poiInfos[p]['category']) for p in t]
                self_similar_revised.append(r)
            for t in other_similar:
                r = [(p, poiInfos[p]['category']) for p in t]
                other_similar_revised.append(r)
            similar['self']=self_similar_revised
            similar['other']=other_similar_revised
        return filtered









