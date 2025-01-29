import logging
import operator

from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()

topics_prompt = "Generate 5 topics of {year} Vietnamese Lunar New Year you want to wish to, output in Vietnamese, separated by commas."
wishes_prompt = "Generate a short wish about this {topic} to wish a happy Lunar new year. Output in Vietnamese"
best_wish_prompt = "Below are a bunch of wishes. Select and return the ID of the best one. \n Wishes: {wishes}"


class Topics(BaseModel):
    topics: list[str]


class Wish(BaseModel):
    wish: str


class BestWish(BaseModel):
    id: int = Field(description="Index of the best wish, starting with 00", ge=0)


class HappyNewYearState(TypedDict):
    year: int
    topics: list[str]
    wishes: Annotated[list, operator.add]
    best_wish: str


class WishState(TypedDict):
    topic: str


model = ChatOllama(model="qwen2.5", temperature=1)


def generate_topics(state: HappyNewYearState):
    response = model.with_structured_output(Topics).invoke(topics_prompt.format(year=state['year']))
    return {"topics": response.topics}


def generate_wish(state: WishState):
    response = model.with_structured_output(Wish).invoke(wishes_prompt.format(topic=state['topic']))
    return {"wishes": [response.wish]}


def select_best_wish(state: HappyNewYearState):
    wishes = "\n\n===============================================\n\n".join(state['wishes'])
    prompt = best_wish_prompt.format(wishes=wishes)
    response = model.with_structured_output(BestWish, method="function_calling").invoke(prompt)
    print(response)
    return {"best_wish": state['wishes'][response.id]}


def continue_to_wishes(state: HappyNewYearState):
    return [Send("generate_wish", {"topic": topic}) for topic in state['topics']]


graph = StateGraph(HappyNewYearState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_wish", generate_wish)
graph.add_node("select_best_wish", select_best_wish)
graph.add_edge("generate_wish", "select_best_wish")
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_wishes, ["generate_wish"])
graph.add_edge("select_best_wish", END)

app = graph.compile()

for s in app.stream({"year": 2025}, stream_mode="values"):
    if 'best_wish' in s:
        print(f'Your best new year wish is: {s['best_wish']}')
