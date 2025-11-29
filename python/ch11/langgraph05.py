import asyncio
from langgraph_sdk import get_client

async def main():
    # langgraph up 호출 시 기본 포트를 변경한 경우에만 url 인자를 get_client()에 전달
    client = get_client(url="http://127.0.0.1:2024")
    # "chatbot_graph"라는 이름으로 배포된 그래프 사용
    assistant_id = "graph6"
    thread = await client.threads.create()

    input = {"messages": [{"role": "user", "content": "미국의 제30대 대통령이 사망했을 때 몇 살이었나요?"}]}
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

if __name__ == "__main__":
    asyncio.run(main())

