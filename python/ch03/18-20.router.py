from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda


'''
datasource 필드에는 'python_docs', 'js_docs' 중 1개의 문자열만 값으로 허용
'''
# 데이터 모델 클래스
class RouteQuery(BaseModel):
    datasource: Literal['python_docs', 'js_docs'] = Field(
        ...,
        description='Given a user question, choose which datasource would be most relevant for answering their question',
    )


# 프롬프트 템플릿
# 함수 호출
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

'''
with_structured_output: 주어진 스키마와 일치하도록 형식화된 출력을 반환하는 모델 래퍼

'''
structured_llm = llm.with_structured_output(RouteQuery)

# 프롬프트
system = '''당신은 사용자 질문을 적절한 데이터 소스로 라우팅하는 전문가입니다. 질문이 지목하는 프로그래밍 언어에 따라 해당 데이터 소스로 라우팅하세요.'''
prompt = ChatPromptTemplate.from_messages(
    [('system', system), ('human', '{question}')]
)

# 라우터 정의
router = prompt | structured_llm

# 실행
question = '''이 코드가 안 돌아가는 이유를 설명해주세요: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(['human', 'speak in {language}']) 
prompt.invoke('french') '''

result = router.invoke({'question': question})
print('\nRouting to: ', result)

def choose_route(result):
    if 'python_docs' in result.datasource.lower():
        return 'chain for python_docs'
    else:
        return 'chain for js_docs'

'''
callable인 경우 (callable() 함수로 확인가능) 아래와 같이 작성 가능

full_chain = router | choose_route
-> 랭체인 LCEL 파이프가 알아서 Runnable로 매핑해준다.
'''
full_chain = router | RunnableLambda(choose_route)

result = full_chain.invoke({'question': question})
print('\nChoose route: ', result)
