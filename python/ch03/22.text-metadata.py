# pip install lark

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

'''
쿼리 구성 : 자연어 쿼리를 사용 중인 데이터베이스 또는 데이터 소스의 쿼리 언어로 변환하는 과정

'''
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'

docs = [
    Document(
        page_content='과학자들이 공룡을 되살리고 대혼란이 일어난다.',
        metadata={'year': 1993, 'rating': 7.7, 'genre': 'SF'},
    ),
    Document(
        page_content='레오나르도 디카프리오가 꿈속의 꿈속의 꿈속의 꿈속에 빠진다.',
        metadata={'year': 2010, 'director': '크리스토퍼 놀란', 'rating': 8.2},
    ),
    Document(
        page_content='심리학자인 형사가 꿈속의 꿈속의 꿈속의 꿈속의 꿈속에 빠진다. 인셉션이 이 발상을 차용했다.',
        metadata={'year': 2006, 'director': '곤 사토시', 'rating': 8.6},
    ),
    Document(
        page_content='평범한 체형의 매우 건강하고 순수한 매력을 지닌 여성들을 남성들이 동경한다.',
        metadata={'year': 2019, 'director': '그레타 거윅', 'rating': 8.3},
    ),
    Document(
        page_content='장난감들이 살아 움직이며 신나는 시간을 보낸다',
        metadata={'year': 1995, 'genre': '애니메이션'},
    ),
    Document(
        page_content='세 남자가 구역으로 들어가고, 세 남자가 구역 밖으로 나온다.',
        metadata={
            'year': 1979,
            'director': '안드레이 타르코프스키',
            'genre': '스릴러',
            'rating': 9.9,
        },
    ),
]

# 문서 임베딩 생성성
embeddings_model = OpenAIEmbeddings()

vectorstore = PGVector.from_documents(
    docs, embeddings_model, connection=connection)

'''
AttributeInfo : LLM이 구조화된 쿼리로 변환하기 위해서 필요한 정보를 담는 클래스
'''
# 쿼리 필드 생성
fields = [
    AttributeInfo(
        name='genre',
        description='영화 장르',
        type='string or list[string]',
    ),
    AttributeInfo(
        name='year',
        description='영화 개봉 연도',
        type='integer',
    ),
    AttributeInfo(
        name='director',
        description='영화 감독',
        type='string',
    ),
    AttributeInfo(
        name='rating',
        description='영화 평점 1-10점',
        type='float',
    ),
]

'''
SelfQueryRetriever : 쿼리를 LLM이 스스로 분석해서 백터 검색용 쿼리, 메타데이터 필터 조건을 자동으로 생성하고, 이를 결합해서 더 정확하게 문서를 검색하는 검색기
fields를 기반으로 구조화된 쿼리를 생성하고 벡터스토어에서 검색을 수행한다.

즉, 쿼리에서 질문과 메타데이터 필터로 사용할 값을 추출 -> fields를 참고하여 구조화된 쿼리 생성 -> 벡터스토어에서 검색

from_llm 메소드를 호출할 때 description은 벡터스토어에 저장된 문서 내용이 뭔지 간략한 설명
'''
description = '영화에 대한 간략한 정보'
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, description, fields)

# 필터 적용
print(retriever.invoke('평점이 8.5점 이상인 영화가 보고 싶어요.'))

print('\n')

# 다양한 필터 적용
print(retriever.invoke(
    '평점이 높은(8.5점 이상) SF영화는 무엇인가요?'))
