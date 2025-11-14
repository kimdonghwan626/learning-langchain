from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(model='gpt-4o', api_key='')
system_msg = SystemMessage(
    '''당신은 문장 끝에 느낌표를 세 개 붙여 대답하는 친절한 어시스턴트입니다.'''
)
human_msg = HumanMessage('프랑스의 수도는 어디인가요?')

response = model.invoke([system_msg, human_msg])
print(response.content)
