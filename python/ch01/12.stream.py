from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


model = ChatOpenAI(model="gpt-3.5-turbo")


template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

'''
제너레이터 yield 키워드가 있으면 여기서 멈춤
함수가 다시 호출되면 다음 값을 반환
스트림을 사용할 때 응답을 순차적으로 표시할 때 유용
'''
@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token

for part in chatbot.stream({
    'question': '거대 언어 모델은 어디서 제공하나요?'
}):
    print(part)
