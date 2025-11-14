from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# 구성 요소

template = ChatPromptTemplate.from_messages(
    [
        ('system', '당신은 친절한 어시스턴트입니다.'),
        ('human', '{question}'),
    ]
)

model = ChatOpenAI(model='gpt-3.5-turbo')

# 함수로 결합한다
# 데코레이터 @chain을 추가해 작성한 함수에 Runnable 인터페이스를 추가한다
'''
@decorator 문법
아래 함수는 다음과 동일
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)
chatbot = chain(chatbot)

chain은 runnable 인터페이스를 구현한 클래스를 반환하기 때문에
chatbot.invoke와 같은 형식으로 사용할 수 있게 된다.
'''
@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


# 사용한다

response = chatbot.invoke({'question': '거대 언어 모델은 어디서 제공하나요?'})
print(response.content)