from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

import os

from ch11.utils import pretty_print, printLog

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
model = ChatOpenAI(model='gpt-4o', api_key=api_key)

question = '안녕하세요? 내 이름이 뭐지?'
context = "내 이름은 홍길동입니다."
userPromptTemplate = '''
다음 Context를 참고하여 Question에 대한 질문을 Answer에 답변해주세요.                                               
Context: {context}
Question: {question}
Answer:'''
chatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "친근한 말투로 답변해주세요."),
    ("user", userPromptTemplate)
])
chatPrompt = chatPromptTemplate.invoke({
    "context" : context,
    "question" : question
})

# response = model.invoke(chatPrompt)
# pretty_print(response)

for token in model.stream(chatPrompt):
    pretty_print(token)