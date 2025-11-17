from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI

# 샘플 메시지 설정
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트입니다.'),
    HumanMessage(content='안녕하세요! 나는 민혁입니다.'),
    AIMessage(content='안녕하세요!'),
    HumanMessage(content='바닐라 아이스크림을 좋아해요.'),
    AIMessage(content='좋네요!'),
    HumanMessage(content='2 + 2는 얼마죠?'),
    AIMessage(content='4입니다.'),
    HumanMessage(content='고마워요.'),
    AIMessage(content='천만에요!'),
    HumanMessage(content='즐거운가요?'),
    AIMessage(content='예!'),
]

# 축약 설정
trimmer = trim_messages(
    max_tokens=65,
    strategy='last', ## 'first' : 최신 메시지 부터 삭제, 'last' : 오래된 메시지부터 삭제
    token_counter=ChatOpenAI(model='gpt-4o-mini'),
    include_system=True, ##시스템 메시지 유지 여부
    allow_partial=False, ## max_tokens을 넘어갈 때, message 통째로 버릴지, message 일부만 자르고 살릴지 여부를 결정
    start_on='human', ## 어떤 메시지부터 잘라내야 할지 결정 'human' : 질문, 답변 삭제, 'assistant' : 답변 삭제
)

# 축약 적용
trimmed = trimmer.invoke(messages)
print(trimmed)
