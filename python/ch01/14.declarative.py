from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 구성 요소

template = ChatPromptTemplate.from_messages(
    [
        ('system', '당신은 친절한 어시스턴트입니다.'),
        ('human', '{question}'),
    ]
)

model = ChatOpenAI()

# 연산자 | 로 결합한다

'''
template | model 는 template.__or__(model) 과 동일, 즉 ChatPromptTemplate.__or__ 메소드를 override 한 것
'''
chatbot = template | model

# 사용한다

response = chatbot.invoke({'question': '거대 언어 모델은 어디서 제공하나요?'})
print(response.content)