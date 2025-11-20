'''
벡터 임베딩 저장을 위한 pgvector 활성화
create extension vector;

문서를 저장할 documents 테이블 생성
create table documents (
    id bigserial primary key,
    content text,   -- Document.pageContent
    metadata jsonb, -- Document.metadata
    embedding vector(1536) -- OpenAI 임베딩
);

문서 검색 함수
create function match_documents (
    query_embedding vector(1536),
    match_count int DEFAULT null,
    filter jsonb DEFAULT '{}'
)
returns table (
    id bigint,
    content text,
    metadata jsonb,
    embedding jsonb,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        id,
        content,
        metadata,
        (embedding::text)::jsonb as embedding,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;
'''

import os
import dotenv

'''
현재 디렉토리에 있는 .env 파일을 읽어서 환경변수로 등록해줌
'''
dotenv.load_dotenv()

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client
from langchain_core.documents import Document

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()

'''
만약 슈파베이스에 등록된 여러 테이블, 함수를 사용하고 싶다면 그 만큼 벡터스토어 인스턴스를 생성하면 됨.
'''
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# 문서 예시
document1 = Document(
    page_content="The powerhouse of the cell is the mitochondria",
    metadata={"source": "https://example.com"}
)

document2 = Document(
    page_content="Buildings are made out of brick", 
    metadata={"source": "https://example.com"}
)

documents = [document1, document2]

# 데이터베이스에 데이터 저장
vector_store.add_documents(documents, ids=["1", "2"])

## 유사도 검색 테스트

query = "biology"
matched_docs = vector_store.similarity_search(query)

print(matched_docs[0].page_content)
