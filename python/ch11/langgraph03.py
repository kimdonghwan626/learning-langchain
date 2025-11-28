import ast
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
from ch11.utils import pretty_print, printLog

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

@tool
def calculator(query: str) -> str:
    '''계산기. 수식만 입력받습니다.'''
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOpenAI(model='gpt-4o', temperature=0.1, api_key=api_key).bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    res = model.invoke(state['messages'])
    return {'messages': res}


builder = StateGraph(State)
builder.add_node('model', model_node)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'model')
builder.add_conditional_edges('model', tools_condition)
builder.add_edge('tools', 'model')

graph = builder.compile()

# 예시
input = {
    'messages': [
        HumanMessage(
            '미국의 제30대 대통령이 사망했을 때 몇 살이었나요?'
        )
    ]
}

'''
stream_mode에 들어갈 값은 다음과 같다.
values : 기본
updates : 변경된 부분만 출력
debug : 이벤트
messages : 메시지만 출력 -> 책에는 안적혀 있는데 되는 것 같음
'''

for c in graph.stream(input, stream_mode='messages'):
    print(c)
