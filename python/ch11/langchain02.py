from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
import os
from pydantic import BaseModel
from ch11.utils import pretty_print, printLog

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model='gpt-4o', api_key=api_key)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')


class RewritedQuery(BaseModel):
    '''재작성한 사용자 질의를 담는 클래스'''
    rewrited: str
    '''재작성한 사용자 질의'''


rewritePromptTemplate = PromptTemplate.from_template('''
당신은 질의 재작성 어시스턴트입니다.

역할:
사용자의 질의를 더 명확하고 구체적으로 정제하여  
LLM 또는 검색 시스템이 더 나은 답변을 생성할 수 있도록 합니다.

재작성 규칙:
- 원래 의미는 그대로 유지합니다.
- 의도를 명확히 하고 부족한 맥락은 자연스럽게 보완합니다.
- “그거”, “저번에 말한…” 같은 불명확한 표현은 구체적으로 바꿉니다.
- 문장을 간결하고 논리적으로 정리합니다.
- 질문에 절대 답변하지 마세요.

사용자 질의:
{query}''')


structured_llm = llm.with_structured_output(RewritedQuery)


rewriteChain = rewritePromptTemplate | structured_llm | StrOutputParser()

query = '''홍길동이라는 사람이 있는 것 같은데 이 사람이 누구인지 설명해줄 수 있나??'''
response = rewriteChain.invoke(query)
print(response)