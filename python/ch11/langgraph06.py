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
    # return eval(query, {"__builtins__": {}}, {})


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

graph = builder.compile()