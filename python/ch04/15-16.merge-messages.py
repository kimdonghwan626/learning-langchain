from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_openai import ChatOpenAI

# 동일한 유형의 연속된 메시지 예제
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트입니다.'),
    SystemMessage(content='항상 농담으로 대답하세요.'),
    HumanMessage(
        content=[{'type': 'text', 'text': '어떤 피자가 제일 맛있나요?'}]
    ),
    HumanMessage(content='어떤 햄버거가 가장 맛있나요?'),
    AIMessage(
        content='나는 항상 너만 "고르곤졸라"'
    ),
    AIMessage(
        content='너가 "버거" 싶어'
    ),
]

'''
-> sys, sys, human, human, ai, ai -> sys, human, ai (개행을 두고 메시지를 병합)
'''
# 연속된 메시지를 병합
merged = merge_message_runs(messages)
print(merged)

# 선언형 구성
model = ChatOpenAI(model='gpt-4o-mini')
merger = merge_message_runs()
chain = merger | model
