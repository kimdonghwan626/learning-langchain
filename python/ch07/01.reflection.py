from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

'''
LangGraph에서 에이전트 정의 : LLM모델이 상태를 가지고 필요한 경우, 툴을 호출하고,
이전 단계의 결과를 보고 스스로 다음 행동을 결정하는 상태 기반 반복 실행 구조
'''

'''
성찰(자기비판) : 생성자와 검토자가 반복적으로 출력과 비판을 반복하여 응답 품질을 높여나간다.
'''
model = ChatOpenAI(model='gpt-4o-mini')

# 상태 타입 정의
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 프롬프트 정의
generate_prompt = SystemMessage(
    '당신은 훌륭한 3단락 에세이를 작성하는 임무를 가진 에세이 어시스턴트입니다.'
    '사용자의 요청에 맞춰 최상의 에세이를 작성하세요.'
    '사용자가 비평을 제공하면, 이전 시도에 대한 수정 버전을 응답하세요.'
)

reflection_prompt = SystemMessage(
    '당신은 에세이 제출물을 평가하는 교사입니다. 사용자의 제출물에 대해 비평과 추천을 생성하세요.'
    '길이, 깊이, 스타일 등과 같은 구체적인 요구사항을 포함한 자세한 추천을 제공하세요.'
)


def generate(state: State) -> State:
    answer = model.invoke([generate_prompt] + state['messages'])
    return {'messages': [answer]}

'''
state['messages'][1:] -> state['messages'][0]를 제외한 나머지
cls_map[msg.__class__](content=msg.content) 
-> msg의 클래스를 키로 map에서 값을 가져오면 반대 클래스가 반환됨
-> 반대 클래스에 msg의 content를 넣어서 새로운 메시지 객체 생성
'''
def reflect(state: State) -> State:
    # 메시지들을 반전시켜 LLM이 자신의 출력을 성찰하도록 합니다.
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    # 첫 번째 메시지는 원래 사용자의 요청입니다. 모든 노드에서 동일하게 유지합니다.
    translated = [reflection_prompt, state['messages'][0]] + [
        cls_map[msg.__class__](content=msg.content) for msg in state['messages'][1:]
    ]
    answer = model.invoke(translated)
    # 이 출력 결과를 생성기(generator)에 대한 사용자 피드백으로 취급합니다.
    return {'messages': [HumanMessage(content=answer.content)]}


def should_continue(state: State):
    if len(state['messages']) > 6:
        # 3회 반복 후, 각 반복마다 2개의 메시지가 쌓이면 종료합니다.
        return END
    else:
        return 'reflect'


# 그래프 구축
builder = StateGraph(State)
builder.add_node('generate', generate)
builder.add_node('reflect', reflect)
builder.add_edge(START, 'generate')
builder.add_conditional_edges('generate', should_continue)
builder.add_edge('reflect', 'generate')

graph = builder.compile()

# 예시
initial_state = {
    'messages': [
        HumanMessage(
            content='오늘날 \'어린 왕자\'가 왜 중요한지에 대해 에세이를 작성하세요.'
        )
    ]
}

'''
아래와 같은 구조로 반환됨
output = {'generate' : {'messaeges' : [...]}}

content[:100] -> 앞에서 100글자만 슬라이싱
'''
# 그래프 실행
for output in graph.stream(initial_state):
    message_type = 'generate' if 'generate' in output else 'reflect'
    print('\nNew message:', output[message_type]
          ['messages'][-1].content[:100], '...')
