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

# 벡터 저장소에서 5개의 관련 문서 검색
retriever = db.as_retriever(search_kwargs={'k': 5})

prompt_rag_fusion = ChatPromptTemplate.from_template(
    '''
하나의 입력 쿼리를 기반으로 여러 개의 검색 쿼리를 생성하는 유용한 어시스턴트입니다.
다음과 관련된 여러 검색 쿼리를 영문으로 생성합니다: 
{question}

출력(쿼리 4개):
''')


def parse_queries_output(message):
    return message.content.split('\n')


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
query_gen = prompt_rag_fusion | llm | parse_queries_output

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'

generated_queries = query_gen.invoke(query)

print('생성된 쿼리: ', generated_queries)


'''
각 쿼리에 대한 관련 문서를 가져와 함수에 전달하여 관련 문서의 최종 목록을 재순위화(즉, 관련성에 따라 순서를 다시 지정)
'''

'''
여러 검색기에서 가져온 문서 순위를 지정한다.
동일한 문서가 많을 수록 점수가 올라간다. 
'''
def reciprocal_rank_fusion(results: list[list], k=60):
    '''여러 순위 문서 목록에 대한 상호 순위 융합 및 RRF 공식에 사용되는 선택적 매개변수 k입니다.'''
    # 사전을 초기화해 각 문서에 대한 융합된 점수를 보관합니다.
    # 고유성을 보장하기 위해 문서가 콘텐츠별로 키로 만듭니다.
    fused_scores = {}
    documents = {}
    for docs in results:
        # 목록에 있는 각 문서를 순위(목록 내 위치)에 따라 반복
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)
    # 융합된 점수를 기준으로 문서를 내림차순으로 정렬하여 최종 재순위 결과를 정리
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]


retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

result = retrieval_chain.invoke(query)

print('순위를 사용해 검색한 컨텍스트: ', result[0].page_content)
print('\n\n')

print('검색한 문서를 바탕으로 응답\n')


prompt = ChatPromptTemplate.from_template(
    '''
다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트:{context}

질문: {question}
'''
)

query = '고대 그리스 철학사의 주요 인물은 누구인가요?'


@chain
def rag_fusion(input):
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke(
        {'context': docs, 'question': input}) 
    answer = llm.invoke(formatted)
    return answer


# 실행
print('RAG 융합 실행\n')
result = rag_fusion.invoke(query)
print(result.content)
