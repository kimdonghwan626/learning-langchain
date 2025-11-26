from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

'''
Annotated : 타입에 메타데이터를 부여하는 용도로 사용, 첫 번째인자는 유효한 타입을 넣어야 한다.
메타데이터는 __metadata__로 접근할 수 있다.

add_messages : Status의 messages 필드를 덮어쓰는게 아닌 추가하는 방식으로 업데이트하라고 LangGraph에게 알려주는 함수
이 함수는 LangGraph 내부에서 노드 실행 후 상태를 병합할 때 자동으로 호출된다.
'''
class State(TypedDict):
    # 메시지의 유형은 list이다. 
    # 어노테이션의 `add_messages` 함수는 상태를 업데이트하는 방법이다. 
    # 이 경우 이전 메시지를 대체하는 대신 새 메시지를 추가한다.
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

model = ChatOpenAI(model='gpt-4o-mini')

def chatbot(state: State):
    answer = model.invoke(state['messages'])
    return {'messages': [answer]}

# 챗봇 노드 추가
# 첫 번째 인자는 고유한 노드 이름
# 두 번째 인자는 실행할 함수 또는 Runnable 
builder.add_node('chatbot', chatbot)

# 엣지 추가
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile()

# 그래프 이미지 저장
graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

# 그래프 실행
input = {'messages': [HumanMessage('안녕하세요!')]}
for chunk in graph.stream(input):
    print(chunk)
