from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    '''사용자의 질문에 대한 답변과 그에 대한 근거(justification)를 함께 제공하세요.'''
    answer: str
    '''사용자의 질문에 대한 답변'''
    justification: str
    '''답변에 대한 근거'''
    count: str

llm = ChatOpenAI(model='gpt-4o', temperature=0, api_key='')
structured_llm = llm.with_structured_output(AnswerWithJustification)

result = structured_llm.invoke('''1 킬로그램의 벽돌과 1 킬로그램의 깃털 중 어느 쪽이 더 무겁나요?''')

print(result.model_dump_json())
