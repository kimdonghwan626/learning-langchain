import asyncio
from contextlib import aclosing

from langchain.schema import HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import ast
from typing import Annotated, TypedDict
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition



@tool
def calculator(query: str) -> str:
    '''계산기. 수식만 입력받습니다.'''
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOpenAI(model='gpt-4o-mini', temperature=0.1).bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    res = model.invoke(state['messages'])
    return {'messages': res}



async def main():
    # 간단한 그래프 생성
    builder = StateGraph(State)
    builder.add_node('model', model_node)
    builder.add_node('tools', ToolNode(tools))
    builder.add_edge(START, 'model')
    builder.add_conditional_edges('model', tools_condition)
    builder.add_edge('tools', 'model')

    # 필요한 노드와 엣지는 여기에 추가
    graph = builder.compile(checkpointer=MemorySaver())

    event = asyncio.Event()

    input = {
        'messages': [
            HumanMessage(
                '미국 제30대 대통령의 사망 당시 나이는 몇 살이었나요?'
            )
        ]
    }
    config = {'configurable': {'thread_id': '1'}}
    
    '''
    output : 비동기 Interator 반환
    
    astream_events : 내부에서 close를 해주기 때문에 aclosing으로 감쌀 필요 없음
    '''
    output = graph.astream_events(input, config, version="v2")

    '''
    async for : 비동기 반복문, 이벤트가 발생할 때마다 event 객체가 yield 됨
    TODO 스트리밍 응답이 어떤 식으로 오는지 확인 필요
    '''
    async for event in output:
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content)


if __name__ == '__main__':
    asyncio.run(main())
