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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import chain

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model='gpt-4o', api_key=api_key)

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
{query}

답변:''')

rewrite_chain = rewritePromptTemplate | llm | StrOutputParser()

loader = TextLoader('ch11\person.txt', encoding='utf-8')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
splitted_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
db = InMemoryVectorStore.from_documents(splitted_docs, embeddings)
retreiver = db.as_retriever(search_kwargs={"k": 3})

promptTemplate = PromptTemplate.from_template('''
다음 Context를 참고하여 사용자 질의에 답변해주세요.
200자 미만으로 답변해주세요.

Context:
{context}

사용자 질의:
{query}

답변:''')

@chain
def llm_chain(query: str):
    printLog(query)
    
    docs = retreiver.invoke(query)
    print(docs)
    
    prompt = promptTemplate.invoke({"context" : docs, "query" : query})
    answer = llm.invoke(prompt)
    return answer

chain = rewrite_chain | llm_chain

query = '''그.. 홍길동이라는 사람이 있는 것 같은데 이 사람이 누구인지 설명해줄 수 있나??'''
result = chain.invoke(query)
printLog(result.content)