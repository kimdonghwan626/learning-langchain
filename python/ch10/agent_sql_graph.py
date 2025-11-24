from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
import json
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
'''
db.dialect : DB 방언, 연결된 DB 종류 및 SQL 문법 정보를 담고 있는 객체
'''
print(db.dialect)
print(db.get_usable_table_names())
'''
db.run을 실행하면 결과는 LLM이 받기 쉽게 문자열로 반환됨
'''
db.run("SELECT * FROM Artist LIMIT 10;")

# gpt4o
llm = ChatOpenAI(model="gpt-4o", temperature=0)
experiment_prefix = "sql-agent-gpt4o"
metadata = "Chinook, gpt-4o agent"

# SQL toolkit
'''
SQLDatabase 객체와 LLM을 바탕으로 에이전트가 SQL 데이터베이스에 접근할 때 필요한 도구들을
자동 생성하는 클래스

get_tools : 테이블 조회, 스키마 조회, 쿼리 실행과 같은 툴 객체 목록을 반환
'''
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

'''
문법에 맞지 않는 쿼리를 입력으로 받은 경우, 수정해서 반환
'''
# Query checking
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Execute the correct query with the appropriate tool."""
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("user", "{query}")])
query_check = query_check_prompt | llm


@tool
def check_query_tool(query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it.
    """
    return query_check.invoke({"query": query}).content

'''
쿼리 결과 평가, 결과가 없는 경우, 다시 시도
'''
# Query result checking
query_result_check_system = """You are grading the result of a SQL query from a DB. 
- Check that the result is not empty.
- If it is empty, instruct the system to re-try!"""
query_result_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_result_check_system), ("user", "{query_result}")])
query_result_check = query_result_check_prompt | llm


@tool
def check_result(query_result: str) -> str:
    """
    Use this tool to check the query result from the database to confirm it is not empty and is relevant.
    """
    return query_result_check.invoke({"query_result": query_result}).content


tools.append(check_query_tool)
tools.append(check_result)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

'''
repr : 파이썬 객체를 문자열로 변환하는 내장 함수
'''
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

'''
LLM이 의미 없는 빈 답을 내면 강제로 재프롬프트해서 실제 출력이 나올 때까지 루프
'''
# Assistant
class Assistant:

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    '''
    객체를 함수처럼 ()로 호출했을 때 실행되는 특수 메소드
    '''
    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # Append to state
            state = {**state}
            # Invoke the tool-calling LLM
            result = self.runnable.invoke(state)
            # If it is a tool call -> response is valid
            # If it has meaninful text -> response is valid
            # Otherwise, we re-prompt it b/c response is not meaninful
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + \
                    [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Assistant runnable
query_gen_system = """
ROLE:
You are an agent designed to interact with a SQL database. You have access to tools for interacting with the database.
GOAL:
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
INSTRUCTIONS:
- Only use the below tools for the following operations.
- Only use the information returned by the below tools to construct your final answer.
- To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.
- Then you should query the schema of the most relevant tables.
- Write your query based upon the schema of the tables. You MUST double check your query before executing it. 
- Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
- You can order the results by a relevant column to return the most interesting examples in the database.
- Never query for all the columns from a specific table, only ask for the relevant columns given the question.
- If you get an error while executing a query, rewrite the query and try again.
- If the query returns a result, use check_result tool to check the query result.
- If the query result result is empty, think about the table schema, rewrite the query, and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")])
assistant_runnable = query_gen_prompt | llm.bind_tools(tools)


# Graph
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))

# Define edges: these determine how the control flow moves
'''
START를 assistant에 연결하는 것과 거의 비슷
'''
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
    # "tools" calls one of our tools. END causes the graph to terminate (and respond to the user)
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
'''
디스크에 저장하지 않고, 메모리에만 저장되는 SQLite DB를 만듬.
'''
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)

'''
입력을 받으면 툴을 사용하여,
-> 쿼리 생성 -> 쿼리 체크
-> 쿼리 실행 -> 결과 체크
-> 응답 체크 후 강제 재실행
을 반복한다.
'''
