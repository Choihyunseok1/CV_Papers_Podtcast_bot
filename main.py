import os
import arxiv
import openai
from notion_client import Client
import datetime

# 1. 깃허브 Secrets에서 열쇠 가져오기
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# 2. 내 정보 설정
GITHUB_USER = "Choihyunseok1"  #본인 ID
GITHUB_REPO = "Paper2Audio"

# 3. 도구(Client) 초기화
notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def run_bot():
    # 저장할 폴더 설정 (절대 경로로 안전하게)
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # 4. arXiv에서 최신 CV 논문 3개 검색
    search = arxiv.Search(
        query="cat:cs.CV", 
        max_results=3, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    for result in search.results():
        # 파일명 생성 (논문 ID 활용)
        paper_id = result.entry_id.split('/')[-1]
        file_name = f"{paper_id}.mp3"
        full_file_path = os.path.join(audio_dir, file_name)

        # 5. GPT-4o에게 팟캐스트 요약 요청
        prompt = f"논문 제목: {result.title}\n초록: {result.summary}\n이 내용을 연구실 등굣길에 듣기 좋은 친근한 한국어 3줄 요약으로 만들어줘."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "너는 IRCV 랩실의 유능한 연구 비서야."},
                      {"role": "user", "content": prompt}]
        )
        summary_text = response.choices[0].message.content

        # 6. OpenAI TTS로 목소리 생성 및 저장
        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=f"안녕하세요! 오늘의 논문 소식입니다. {result.title}에 대해 알려드릴게요. {summary_text}"
        )
        audio_response.stream_to_file(full_file_path)

        # 7. 노션에 등록할 공개 URL 주소 생성
        audio_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name}"

        # 8. 노션 페이지 생성
        notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "이름": {"title": [{"text": {"content": result.title}}]},
                "날짜": {"date": {"start": datetime.date.today().isoformat()}},
                "카테고리": {"select": {"name": "CV"}}, # 노션에서 '선택' 유형이어야 함
                "요약": {"rich_text": [{"text": {"content": summary_text}}]},
                "오디오": {"url": audio_url}
            }
        )
        print(f"처리 완료: {result.title}")

if __name__ == "__main__":
    run_bot()
