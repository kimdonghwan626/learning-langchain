from agent_sql_graph import builder
from langchain import hub
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from langchain_core.runnables import Runnable
from agent_sql_graph import assistant_runnable
import uuid
from dotenv import load_dotenv

load_dotenv()

_printed = set()
thread_id = str(uuid.uuid4())
experiment_prefix = "sql-agent-gpt4o"
metadata = "chinook-gpt-4o-base-case-agent"
config = {
    "configurable": {
        # 체크포인트는 thread_id로 접근합니다
        "thread_id": thread_id,
    }
}


def predict_sql_agent_answer(example: dict):
    """답변 평가에 사용합니다"""
    msg = {"messages": ("user", example["input"])}
    messages = graph.invoke(msg, config)
    return {"response": messages['messages'][-1].content}


# 평가 프롬프트
grade_prompt_answer_accuracy = hub.pull(
    "langchain-ai/rag-answer-vs-reference")


def answer_evaluator(run, example) -> dict:
    """
    RAG 답변 정확도를 위한 간단한 평가기
    """

    # 질문, 정답, 체인 답변 가져오기
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs["response"]

    # LLM 평가기
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 구조화된 프롬프트
    answer_grader = grade_prompt_answer_accuracy | llm

    # 평가기 실행
    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}


dataset_name = "sql-agent-response"
experiment_results = evaluate(
    predict_sql_agent_answer,
    data=dataset_name,
    evaluators=[answer_evaluator],
    num_repetitions=3,
    experiment_prefix=experiment_prefix,
    metadata={"version": metadata},
)


"""
단일 툴 평가
"""


def predict_assistant(example: dict):
    """단일 툴 호출 평가를 위한 어시스턴트 호출"""
    msg = [("user", example["input"])]
    result = assistant_runnable.invoke({"messages": msg})
    return {"response": result}


def check_specific_tool_call(root_run: Run, example: Example) -> dict:
    """
    응답의 첫 번째 툴 호출이 예상된 툴 호출과 일치하는지 확인합니다.
    """

    # 예상되는 툴 호출
    expected_tool_call = 'sql_db_list_tables'

    # 실행
    response = root_run.outputs["response"]

    # 툴 호출 가져오기
    try:
        tool_call = getattr(response, 'tool_calls', [])[0]['name']

    except (IndexError, KeyError):
        tool_call = None

    score = 1 if tool_call == expected_tool_call else 0
    return {"score": score, "key": "single_tool_call"}


experiment_results = evaluate(
    predict_assistant,
    data=dataset_name,
    evaluators=[check_specific_tool_call],
    experiment_prefix=experiment_prefix + "-single-tool",
    num_repetitions=3,
    metadata={"version": metadata},
)


"""
에이전트 진행 과정 평가
"""

def predict_sql_agent_messages(example: dict):
    """답변 평가에 사용됩니다"""
    msg = {"messages": ("user", example["input"])}
    graph = builder.compile()
    messages = graph.invoke(msg, config)
    return {"response": messages}


def find_tool_calls(messages):
    """
    반환된 메시지에서 모든 툴 호출 찾기
    """
    tool_calls = [tc['name'] for m in messages['messages']
                  for tc in getattr(m, 'tool_calls', [])]
    return tool_calls

'''
evaluator 함수는 파라미터가 정해져 있음 (run기반, dict기반)
* dict기반은 agent_evaluation_rag.py 참고
'''
def contains_all_tool_calls_any_order(root_run: Run, example: Example) -> dict:
    """
    예상되는 모든 툴이 순서에 관계없이 호출되었는지 확인합니다.
    """
    expected = ['sql_db_list_tables', 'sql_db_schema',
                'sql_db_query_checker', 'sql_db_query', 'check_result']
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(messages)
    # 모든 툴 호출 출력
    # print("Here are my tool calls:")
    # print(tool_calls)
    if set(expected) <= set(tool_calls):
        score = 1
    else:
        score = 0
        
    '''
    evaluator 함수 반환값은 
    run 기반인 경우, dict
    dict 기반인 경우, bool, int, float, dict 중 하나로 반환되어야 한다.
    '''
    return {"score": int(score), "key": "multi_tool_call_any_order"}


def contains_all_tool_calls_in_order(root_run: Run, example: Example) -> dict:
    """
    예상되는 모든 툴이 정확한 순서로 호출되었는지 확인합니다.
    """
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(messages)
    # 모든 툴 호출 출력
    # print("Here are my tool calls:")
    # print(tool_calls)
    it = iter(tool_calls)
    expected = ['sql_db_list_tables', 'sql_db_schema',
                'sql_db_query_checker', 'sql_db_query', 'check_result']
    if all(elem in it for elem in expected):
        score = 1
    else:
        score = 0
        
    return {"score": int(score), "key": "multi_tool_call_in_order"}


def contains_all_tool_calls_in_order_exact_match(root_run: Run, example: Example) -> dict:
    """
    예상되는 모든 툴이 정확한 순서로 호출되고 추가 툴 호출이 없는지 확인합니다.
    """
    expected = ['sql_db_list_tables', 'sql_db_schema',
                'sql_db_query_checker', 'sql_db_query', 'check_result']
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(messages)
    # 모든 툴 호출 출력
    # print("Here are my tool calls:")
    # print(tool_calls)
    if tool_calls == expected:
        score = 1
    else:
        score = 0

    return {"score": int(score), "key": "multi_tool_call_in_exact_order"}


experiment_results = evaluate(
    predict_sql_agent_messages,
    data=dataset_name,
    evaluators=[contains_all_tool_calls_any_order, contains_all_tool_calls_in_order,
                contains_all_tool_calls_in_order_exact_match],
    experiment_prefix=experiment_prefix + "-trajectory",
    num_repetitions=3, ## 반복 횟수
    metadata={"version": metadata},
)
