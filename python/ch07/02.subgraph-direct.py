from typing import TypedDict
from langgraph.graph import START, StateGraph

'''
부모 그래프에서 서브그래프 직접 호출하는 예제

'''
# 부모 그래프와 서브그래프에서 사용할 상태
class State(TypedDict):
    foo: str  # 서브그래프와 이 키를 공유


class SubgraphState(TypedDict):
    foo: str  # 부모 그래프와 이 키를 공유
    bar: str


# 서브그래프 정의
def subgraph_node(state: SubgraphState):
    # 서브그래프 노드는 공유 키인 "foo"를 사용해 부모 그래프와 통신한다
    return {"foo": state["foo"] + "bar"}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")
# 서브그래프에 필요한 추가 설정은 여기에 작성
subgraph = subgraph_builder.compile()

# 부모 그래프 정의
builder = StateGraph(State)
builder.add_node("subgraph", subgraph)
builder.add_edge(START, "subgraph")
# 부모 그래프에 필요한 추가 설정은 여기에 작성
graph = builder.compile()

# 예시
initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)
print(f"Result: {result}")  # foo에 "bar"가 추가되어야 함
