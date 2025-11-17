'''
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. pip install -qU langchain_postgres
3. Run the following command to start the postgres container:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
4. Use the connection string below for the postgres container

'''

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain


connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

# 문서를 로드 후 분할
raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# 문서에 대한 임베딩 생성
embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

# 벡터 저장소에서 2개의 관련 문서 검색
retriever = db.as_retriever(search_kwargs={'k': 2})

# 관련 질문을 하기 전에 관련 없는 정보로 시작하는 쿼리
query = '일어나서 이를 닦고 뉴스를 읽었어요. 그러다 전자레인지에 음식을 넣어둔 걸 깜빡했네요. 고대 그리스 철학사의 주요 인물은 누구인가요?'

# 관련 문서 받아오기
docs = retriever.invoke(query)

print(docs[0].page_content)
print('\n\n')

prompt = ChatPromptTemplate.from_template(
    '''다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트:{context}

질문: {question}
'''
)

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)


# 이번에는 효율성을 위해 로직을 캡슐화한 후 재실행.

# 데코레이터 @chain는 이 함수를 LangChain runnable로 변환하여
# LangChain의 chain 작업과 파이프라인과 호환되게 함

@chain
def qa(input):
    # 관련 문서 검색
    docs = retriever.invoke(input)
    # 프롬프트 포매팅
    formatted = prompt.invoke({'context': docs, 'question': input})
    # 답변 생성
    answer = llm.invoke(formatted)
    return answer


# 실행
result = qa.invoke(query)
print(result.content)


# 쿼리를 재작성하여 정확도를 높이기
print('\n정확도를 높이기 위해 쿼리를 재작성\n')

rewrite_prompt = ChatPromptTemplate.from_template(
    '''
웹 검색 엔진이 주어진 질문에 답할 수 있도록 더 나은 영문 검색어를 제공하세요. 쿼리는 \'**\'로 끝내세요. 

질문: {x} 

답변:
''')

'''
strip 양쪽 끝에 있는 특정 문자를 제거
예) "'message'"** -> "message"** -> "message"

@chain을 붙이지 않아도 | 다음에 일반 파이썬 함수를 추가하는 경우, 자동으로 Runnable로 감싸줌
'''
def parse_rewriter_output(message):
    return message.content.strip('\'').strip('**')


rewriter = rewrite_prompt | llm | parse_rewriter_output


@chain
def qa_rrr(input):
    # 쿼리 재작성
    new_query = rewriter.invoke(input)
    print('재작성한 쿼리: ', new_query)
    # 관련 문서 검색
    docs = retriever.invoke(new_query)
    # 프롬프트 포매팅
    formatted = prompt.invoke({'context': docs, 'question': input})
    # 답변 생성
    answer = llm.invoke(formatted)
    return answer


print('\n재작성한 쿼리로 모델 호출\n')

# 재작성한 쿼리로 재실행
result = qa_rrr.invoke(query)
print(result.content)
