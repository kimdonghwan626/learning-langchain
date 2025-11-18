from typing import TypedDict
from langgraph.graph import START, StateGraph

'''
서브 그래프와 스키마가 일치하지 않아 서브 그래프의 입출력에 함수를 두는 예제
'''

class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    # 부모 그래프와 키를 공유하지 않음
    bar: str
    baz: str


# 서브그래프 정의
def subgraph_node(state: SubgraphState):
    return {"bar": state["bar"] + "baz"}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")
# 서브그래프에 필요한 추가 설정은 여기에 작성
subgraph = subgraph_builder.compile()


# 서브그래프를 호출하는 부모 그래프 정의
def node(state: State):
    # 부모 그래프의 상태를 서브그래프 상태로 변환
    response = subgraph.invoke({"bar": state["foo"]})
    # 응답을 다시 부모 그래프의 상태로 변환
    return {"foo": response["bar"]}


builder = StateGraph(State)
# 서브그래프 대신 `node`를 지정
builder.add_node("node", node)
builder.add_edge(START, "node")
# 부모 그래프에 필요한 추가 설정은 여기에 작성
graph = builder.compile()

# 예시
initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)
print(
    f"Result: {result}"
)  # foo를 bar로 변환해 "baz"를 추가하고 다시 foo로 변환
