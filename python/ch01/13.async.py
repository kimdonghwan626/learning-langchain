from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(model='gpt-4o', temperature=0, api_key='')


@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)


async def main():
    return await chatbot.ainvoke({'question': '거대 언어 모델은 어디서 제공하나요?'})

'''
__name__ : 이 파일을 직접 실행한 경우 __main__이 반환됨. 즉 이 파일을 직접 실행한 경우에만 아래 로직을 실행하도록 함.
asyncio : 파이썬은 비동기 함수를 직접 실행할 수 없고 asyncio라는 이벤트 루프를 만들고 코루틴 객체(async 함수)를 던져서 실행한다.
'''
if __name__ == "__main__":
    import asyncio
    print(asyncio.run(main()))
