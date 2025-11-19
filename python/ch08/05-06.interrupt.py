import asyncio
from contextlib import aclosing

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

@tool
def calculator(query: str) -> str:
    '''계산기. 수식만 입력받습니다.'''
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={'name': tool.name}) for tool in tools],
    embeddings,
).as_retriever()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state['selected_tools']]
    res = model.bind_tools(selected_tools).invoke(state['messages'])
    return {'messages': res}


def select_tools(state: State) -> State:
    query = state['messages'][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {'selected_tools': [doc.metadata['name'] for doc in tool_docs]}




async def main():
    # 6.4절 아키텍처 이용
    builder = StateGraph(State)
    builder.add_node('select_tools', select_tools)
    builder.add_node('model', model_node)
    builder.add_node('tools', ToolNode(tools))
    builder.add_edge(START, 'select_tools')
    builder.add_edge('select_tools', 'model')
    builder.add_conditional_edges('model', tools_condition)
    builder.add_edge('tools', 'model')

    '''
    사용자 개입 패턴을 사용하려면 체크포인터가 반드시 있어야 한다.
    '''
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
    
    # 중단 태스크 생성
    '''
    interrupt(event) 코루틴을 Task 객체로 감싸서 Event Loop에 등록
    -> 백그라운드에서 돌 준비가 된 상태
    -> 실제 실행은 다음 await 지점들에서 event loop가 돌 때마다 조금씩 진행
    
    await something() 이라고 되어 있는 경우, somthing 내부 await를 만나기 전까지 실행 후,
    event loop에게 제어권을 넘김
    '''
    interrupt_task = asyncio.create_task(interrupt(event))

    '''
    aclosing : async iterator를 사용할 때 예외가 발생한 경우, 
    리소스 누수가 발생하지 않도록 자동으로 aclose 호출
    
    async with : 준비 __anter__(), 정리 __aexit__() 메소드를 자동 호출,
    async iterator aclose를 호출하지 않기 때문에 aclosing과 같이 사용해야 한다.
    '''
    async with aclosing(graph.astream(input, config)) as stream:
        async for chunk in stream:
            '''
            Event 내부 플래그가 True 인지 확인, True인 경우, 답변 출력을 중단 
            '''
            if event.is_set():
                break
            else:
                print(chunk)  # 출력

    '''
    아래 코드를 명시하지 않아도 동작은 하지만 interrupt 함수에서 예외가 발생해도
    main 함수가 catch를 하지 못함 
    
    asyncio.run은 main이 끝나면 남아있는 Task들을 cancel하고 정리
    await를 명시하면 interrupt가 완전히 끝날 때까지 기다린 후 main을 종료
    '''
    await interrupt_task
    
    
async def interrupt(event):
    # 2초 후 중단
    await asyncio.sleep(2)
    '''
    Event 내부 플래그를 True로 바꿈 
    -> 이 이벤트를 기다리고 있던(wait 중인) 태스크 들을 한 번에 깨움
    
    event.wait() : event가 set될 때 까지 대기, set이 되면 바로 다음 코드 진행
    '''
    event.set()
    print("중단 신호를 보냈습니다.")

if __name__ == '__main__':
    asyncio.run(main())