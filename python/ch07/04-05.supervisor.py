from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from pydantic import BaseModel

'''
멀티 에이전트 아키텍처
-> 슈퍼바이저 아키텍처 : 관리자가 중간에서 어떤 에이전트를 호출할 지 결정
'''
class SupervisorDecision(BaseModel):
    next: Literal['researcher', 'coder', 'FINISH']


# 모델 초기화
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
model = model.with_structured_output(SupervisorDecision)

# 사용 가능한 에이전트 정의
agents = ['researcher', 'coder']

# 시스템 프롬프트 정의
system_prompt_part_1 = f'''당신은 다음 서브에이전트 사이의 대화를 관리하는 슈퍼바이저입니다. 서브에이전트: {agents}. 아래 사용자 요청에 따라,  
다음으로 행동할 서브에이전트를 지목하세요. 각 서브에이전트는 임무를 수행하고 결과와 상태를 응답합니다. 실행할 서브에이전트가 없거나 작업이 완료되면,  
FINISH로 응답하세요.'''

system_prompt_part_2 = f'''위 대화를 바탕으로, 다음으로 행동할 서브에이전트는 누구입니까? 아니면 FINISH 해야 합니까? 서브에이전트: {', '.join(agents)}, FINISH'''


def supervisor(state):
    messages = [
        ('system', system_prompt_part_1),
        *state['messages'],
        ('system', system_prompt_part_2),
    ]
    return model.invoke(messages)


# 에이전트 상태 정의
class AgentState(MessagesState):
    next: Literal['researcher', 'coder', 'FINISH']


# 에이전트 함수 정의
def researcher(state: AgentState):
    # 실제 구현에서는 이 함수가 리서치 작업을 수행합니다.
    # 여기서는 임의로 관련 데이터를 찾는 척 합니다.
    response = {
        'role': 'assistant',
        'content': '관련 데이터를 찾는 중입니다... 잠시만 기다려주세요.',
    }
    # 임의의 데이터 생성
    fake_data = {
        'data': '전세계 인구 데이터: [미국: 331M, 중국: 1.4B, 인도: 1.3B]'
    }
    response['content'] += f'\n찾은 데이터: {fake_data['data']}'
    return {'messages': [response]}

def coder(state: AgentState):
    # 실제 구현에서는 이 함수가 코드를 작성합니다.
    # 여기서는 임의로 코드를 작성하는 척 합니다.
    response = {
        'role': 'assistant',
        'content': '코드를 작성 중입니다... 잠시만 기다려주세요.',
    }
    # 임의의 코드 생성
    fake_code = '''
def visualize_population(data):
    import matplotlib.pyplot as plt

    countries = list(data.keys())
    population = list(data.values())

    plt.bar(countries, population)
    plt.xlabel('Country')
    plt.ylabel('Population')
    plt.title('World Population by Country')
    plt.show()

data = {'USA': 331, 'China': 1400, 'India': 1300}
visualize_population(data)
'''
    response['content'] += f'\n작성된 코드:\n{fake_code}'
    return {'messages': [response]}


# 그래프 구축
builder = StateGraph(AgentState)
builder.add_node('supervisor', supervisor)
builder.add_node('researcher', researcher)
builder.add_node('coder', coder)

builder.add_edge(START, 'supervisor')
# 슈퍼바이저의 결정에 따라 에이전트 중 하나로 라우팅하거나 종료합니다.
builder.add_conditional_edges('supervisor', lambda state: state['next'])
builder.add_edge('researcher', 'supervisor')
builder.add_edge('coder', 'supervisor')

graph = builder.compile()

# 예시
initial_state = {
    'messages': [
        {
            'role': 'user',
            'content': '전세계 인구를 국적을 기준으로 시각화 해주세요.',
        }
    ],
    'next': 'supervisor',
}

'''
graph.invoke는 최종 결과 state만 반환한다.
graph.stream을 호출하면 각 노드의 출력을 딕셔너리 형태로 yield한다. (model의 stream과 다름)

d = {"a": 1, "b": 2}
-> d.items() -> dict_items([('a', 1), ('b', 2)]) (dict_items : iterable 객체)
-> list(d.items()) -> [('a', 1), ('b', 2)]
'''
for output in graph.stream(initial_state):
    node_name, node_result = list(output.items())[0]
    print(f'\n현재 노드: {node_name}')
    if node_result.get('messages'):
        print(f'응답: {node_result['messages'][-1]['content'][:100]}...')
    print(f'\n다음 단계: {node_result.get('next', 'N/A')}')