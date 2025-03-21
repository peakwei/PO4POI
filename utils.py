from math import radians, sin, cos, sqrt, atan2
from typing import Optional
from openai import AsyncOpenAI,OpenAIError
import asyncio
import random
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1 = eval(lat1)
    lon1 = eval(lon1)
    lat2 = eval(lat2)
    lon2 = eval(lon2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return int(distance)

def create_async_client(key, base):
    client = AsyncOpenAI(
        api_key=key,
        base_url=base)
    return client

