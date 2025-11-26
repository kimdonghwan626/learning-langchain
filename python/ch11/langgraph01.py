from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
import os
from pydantic import BaseModel
from ch11.utils import pretty_print, printLog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import chain
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model='gpt-4o', api_key=api_key)

class State(TypedDict):
    
    messages: Annotated[list, add_messages]
    
    
def chatbot(state: State):
    answer = llm.invoke(state["messages"])
    
    printLog(type(answer))
    
    return {"messages" : [answer]}
        
builder = StateGraph(State)
builder.add_node('chatbot', chatbot)
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile(checkpointer = MemorySaver())

input = {"messages" : [HumanMessage("안녕하세요! 저는 홍길동입니다.")]}
thread1 = {"configurable" : {"thread_id" : "1"}}


for chunk in graph.stream(input = input, config = thread1):
    print(type(chunk))
    print(chunk)

state = graph.get_state(thread1)
printLog(type(state))
printLog(state)

graph.update_state(thread1, {'messages': [AIMessage('저는 LLM이 좋아요!')]})
state = graph.get_state(thread1)
printLog(state)