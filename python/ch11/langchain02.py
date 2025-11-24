from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    '''사용자의 질문에 대한 답변과 그에 대한 근거(justification)를 함께 제공하세요.'''
    answer: str
    '''사용자의 질문에 대한 답변'''
    justification: str
    '''답변에 대한 근거'''

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model='gpt-4o', api_key=api_key, temperature=0)

structured_llm = llm.with_structured_output(AnswerWithJustification)

result = structured_llm.invoke('''1 킬로그램의 벽돌과 1 킬로그램의 깃털 중 어느 쪽이 더 무겁나요?''')

from ch11.utils import pretty_print, printLog
pretty_print(result)
