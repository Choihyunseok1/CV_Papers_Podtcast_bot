import os
import arxiv
import openai
from notion_client import Client
import datetime

# 깃허브 금고에서 열쇠 가져오기
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def run_bot():
    # 1. 최신 논문 검색
    search = arxiv.Search(query="cat:cs.CV", max_results=3, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    for result in search.results():
        # 파일명을 안전하게 만들기 (특수문자 제거)
        safe_title = "".join(x for x in result.entry_id.split('/')[-1] if x.isalnum())
        file_path = f"{safe_title}.mp3"

        # 2. GPT로 팟캐스트 대본 작성
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "너는 IRCV 랩실의 AI 요약 비서야. 논문을 등굣길 팟캐스트처럼 친근하게 3줄 요약해줘."},
                      {"role": "user", "content": f"제목: {result.title}\n초록: {result.summary}"}]
        )
        summary_text = response.choices[0].message.content

        # 3. OpenAI TTS로 목소리 생성
        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", # 목소리 타입 (alloy, echo, fable, onyx, nova, shimmer)
            input=f"안녕하세요! 오늘의 논문 요약입니다. 제목은 {result.title}입니다. {summary_text}"
        )
        audio_response.stream_to_file(file_path)

        # 4. 노션에 등록 (우선 텍스트 위주로 등록)
        notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "이름": {"title": [{"text": {"content": result.title}}]},
                "날짜": {"date": {"start": datetime.date.today().isoformat()}},
                "카테고리": {"select": {"name": "CV"}},
                "요약": {"rich_text": [{"text": {"content": summary_text}}]},
                "오디오": {"url": result.pdf_url} # 실제 MP3 링크는 2단계에서 해결!
            }
        )
        print(f"오디오 생성 및 등록 완료: {result.title}")

if __name__ == "__main__":
    run_bot()
