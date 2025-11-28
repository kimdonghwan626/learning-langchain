import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]
    
@tool
def calculator(query: str) -> str:
    '''계산기. 수식만 입력받습니다.'''
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=api_key)

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={'name': tool.name}) for tool in tools],
    embeddings,
).as_retriever()

def select_tools(state: State) -> State:
    query = state['messages'][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {'selected_tools': [doc.metadata['name'] for doc in tool_docs]}

model = ChatOpenAI(model='gpt-4o', temperature=0.1, api_key=api_key)

def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state['selected_tools']]
    res = model.bind_tools(selected_tools).invoke(state['messages'])
    return {'messages': res}

builder = StateGraph(State)
builder.add_node('select_tools', select_tools)
builder.add_node('model', model_node)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'select_tools')
builder.add_edge('select_tools', 'model')
builder.add_conditional_edges('model', tools_condition)
builder.add_edge('tools', 'model')

graph = builder.compile(checkpointer=MemorySaver())

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

input = {
    'messages': [
        HumanMessage(
            '미국 제30대 대통령의 사망 당시 나이는 몇 살이었나요?'
        )
    ]
}

config = {"configurable": {"thread_id": "1"}}

output = graph.stream(input, config, interrupt_before=["model"])

for c in output:
    print(c)

print("--------------------중단")

state = graph.get_state(config)
print(state)

print("--------------------업데이트")

# 첫 번째 메시지의 content를 변경하기 위한 업데이트
updated_messages = state.values['messages'].copy()
updated_messages[0] = HumanMessage(content="영화 보이후드의 촬영 기간은 얼마나 되나요?")

# 상태 업데이트 적용
'''
update_state는 그래프를 실행하지 않고 state 값만 변경
'''
update = {'messages': updated_messages}
graph.update_state(config, update)
state = graph.get_state(config)
print(state)

print("--------------------재개")

# 업데이트된 상태로 그래프 계속 실행
output = graph.stream(None, config)
for c in output:
    print(c)
    
print("--------------------히스토리")
history = [state for state in graph.get_state_history(config)]
print(history)

output = graph.stream(None, history[2].config)
for c in output:
    print(c)