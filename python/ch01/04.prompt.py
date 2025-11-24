from langchain_core.prompts import PromptTemplate

'''
PromptTemplate는 Human 메시지만 생성
ChatPromptTemplate는 System, Human, Assistant 생성
'''
template = PromptTemplate.from_template('''아래 작성한 컨텍스트(Context)를 기반으로
    질문(Question)에 대답하세요. 제공된 정보로 대답할 수 없는 질문이라면 "모르겠어요." 라고 답하세요.

Context: {context}

Question: {question}

Answer: ''')

response = template.invoke({
    'context': '''거대 언어 모델(LLM)은 자연어 처리(NLP) 분야의 최신 발전을 이끌고 있습니다.
    거대 언어 모델은 더 작은 모델보다 우수한 성능을 보이며, NLP 기능을 갖춘 애플리케이션을 개발하는 개발자들에게
    매우 중요한 도구가 되었습니다. 개발자들은 Hugging Face의 `transformers` 라이브러리를
    활용하거나, `openai` 및 `cohere` 라이브러리를 통해 OpenAI와 Cohere의 서비스를 이용하여
    거대 언어 모델을 활용할 수 있습니다.
    ''',
    'question': '거대 언어 모델은 어디서 제공하나요?'
})

print(response)
