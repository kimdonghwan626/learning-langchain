from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

'''
실험 평가 데이터셋 관리 플랫폼에 접근하기 위한 클라이언트
환경 변수에 저장된 LANGSMITH_API_KEY를 기반으로 랭스미스 서버에 연결
'''
client = Client()

# 데이터셋 제작
examples = [
    ("어느 나라의 고객이 가장 많이 지출했나요? 그리고 얼마를 지출했나요?",
     "가장 많이 지출한 나라는 미국으로, 총 지출액은 $523.06입니다"),
    ("2013년에 가장 많이 판매된 트랙은 무엇인가요?",
     "2013년에 가장 많이 판매된 트랙은 Hot Girl입니다."),
    ("Led Zeppelin 아티스트는 몇 개의 앨범을 발매했나요?",
     "Led Zeppelin은 14개의 앨범을 발매했습니다"),
    ("'Big Ones' 앨범의 총 가격은 얼마인가요?",
     "'Big Ones' 앨범의 총 가격은 14.85입니다"),
    ("2009년에 어떤 영업 담당자가 가장 많은 매출을 올렸나요?",
     "Steve Johnson이 2009년에 가장 많은 매출을 올렸습니다"),
]

'''
클라이언트에 동일한 이름의 데이터 셋이 있는지 확인하고 없으면 생성
'''
dataset_name = "sql-agent-response"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    inputs, outputs = zip(
        *[({"input": text}, {"output": label}) for text, label in examples]
    )
    client.create_examples(
        inputs=inputs, outputs=outputs, dataset_id=dataset.id)
