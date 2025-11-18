import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

'''
@tool : 함수가 LangChain Tool 객체로 변환하게 해주는 데코레이터
1) LLM이 툴을 호출할 때 Json 스키마를 자동으로 생성
2) 함수 이름을 tool name으로 등록
3) 입출력 타입 검증
-> LangGraph 노드에서 호출 가능
LLM이 이 함수를 자동으로 호출하게 만들려면 반드시 필요함
'''
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


'''
ToolNode : LangChain 툴을 그래프에서 자동 실행해주는 에이전트 엔진 역할
tools_condition : tool call 메시지를 생성했는지 여부를 감지하는 조건 함수
-> LLM이 tool 호출을 요청한 경우 ToolNode로 이동
-> 그렇지 않은 경우, LLM 종료 또는 후속 처리
tools_condition은 조건에 따라 다음에 실행할 노드 이름을 반환한다. (tools or __END__)
만약 ToolNode를 'tools2'로 등록한 경우, tools_condition 함수를 재정의해야 한다.
'''
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

for c in graph.stream(input):
    print(c)
