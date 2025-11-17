'''
이번 예시에서는 sqlite를 통해 Chinook_Sqlite.sql를 사용한다. 실습 실행 전에 sqlite와 sql 파일을 다운받아야 한다.

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

`Chinook.db`를 코드를 실행할 디렉터리에 옮긴다.

'''

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

# 사용할 db 경로로 수정
db = SQLDatabase.from_uri('sqlite:///Chinook.db')
print(db.get_usable_table_names())

llm = ChatOpenAI(model='gpt-4o', temperature=0)

'''
create_sql_query_chain : db 스키마 정보를 스캔, 쿼리를 분석하여 스키마에 맞는 SQL쿼리를 생성하여 반환해준다.
'''
# 질문을 SQL 쿼리로 변환
write_query = create_sql_query_chain(llm, db)

'''
QuerySQLDatabaseTool : SQL 쿼리를 받아서 DB를 조회하고 결과를 파이썬 객체(list, dict)로 반환
'''
# SQL 쿼리 실행
execute_query = QuerySQLDatabaseTool(db=db)

# combined chain = write_query | execute_query
combined_chain = write_query | execute_query

# 체인 실행
result = combined_chain.invoke({'question': '직원(employee)은 모두 몇 명인가요?'})

print(result)
