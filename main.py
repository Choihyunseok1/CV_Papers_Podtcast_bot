import os
import arxiv
import openai
from notion_client import Client
import datetime
from pytz import timezone

# 1. 설정값 가져오기
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GITHUB_USER = "Choihyunseok1"
GITHUB_REPO = "Paper2Audio"

notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def run_bot():
    # 저장 경로 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # 2. 시간대 설정 (서울 시간 기준 어제 18:00 ~ 현재)
    seoul_tz = timezone('Asia/Seoul')
    now = datetime.datetime.now(seoul_tz)
    yesterday_6pm = (now - datetime.timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

    # 3. arXiv 논문 검색 (최근 등록 순으로 10개까지 가져와서 시간 필터링)
    search = arxiv.Search(
        query="cat:cs.CV",
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    valid_papers = []
    for p in search.results():
        # arXiv의 published 시간은 UTC 기준이므로 서울 시간으로 변환하여 비교
        p_date = p.published.astimezone(seoul_tz)
        if p_date > yesterday_6pm:
            valid_papers.append(p)

    if not valid_papers:
        print("해당 시간대에 새로 올라온 논문이 없습니다.")
        return

    # 4. 모든 논문을 하나의 프롬프트로 통합
    papers_info = ""
    paper_titles_list = []
    for i, p in enumerate(valid_papers):
        papers_info += f"논문 {i+1} 제목: {p.title}\n초록: {p.summary}\n\n"
        paper_titles_list.append(p.title)

    combined_prompt = f"""
    아래는 어제 저녁부터 오늘 새벽 사이에 새로 발표된 {len(valid_papers)}개의 컴퓨터 비전 논문입니다.
    
    {papers_info}

    위 논문들을 바탕으로 다음 두 가지를 작성해 주세요.

    1. [요약]
    - 노션 기록용 핵심 요약.
    - 각 논문별로 제목을 언급하고, '-함', '-임' 형태의 짧은 요약체로 2~3줄씩 작성.
    - 논문들 사이는 줄바꿈으로 구분할 것.

    2. [대본]
    - 형식: 라디오 방송 '모닝 AI 브리핑' 통합 스크립트.
    - {len(valid_papers)}개의 논문을 자연스럽게 연결하며 하나의 에피소드로 구성할 것.
    - 도입부에서 오늘 브리핑할 논문 개수를 언급하며 시작.
    - 차분하고 정중한 어조 유지.
    - 모든 영어 약어와 고유 명사는 TTS가 오독하지 않도록 한글 발음대로 표기해 주세요. (예: IRCV -> 아이알씨브이, SOTA -> 소타)
    - 마무리 멘트와 함께 정중한 인사.
    - 논문 하나당 1~2분의 길이로 설명할 것.

    출력 형식:
    [요약]
    (통합 요약 내용)

    [대본]
    (통합 라디오 스크립트 내용)
    """

    # 5. GPT-4o에게 통합 요청
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "너는 IRCV 랩실의 수석 연구 비서야."},
                  {"role": "user", "content": combined_prompt}]
    )
    full_text = response.choices[0].message.content
    summary_text = full_text.split("[대본]")[0].replace("[요약]", "").strip()
    audio_script = full_text.split("[대본]")[1].strip()

    # 6. 하나의 통합 오디오 파일 생성
    today_str = now.strftime("%Y%m%d")
    file_name = f"Daily_Briefing_{today_str}.mp3"
    full_file_path = os.path.join(audio_dir, file_name)

    audio_response = client.audio.speech.create(
        model="tts-1-hd",
        voice="echo",
        input=audio_script,
        speed=0.8
    )
    audio_response.stream_to_file(full_file_path)

    # 7. 노션에 단 하나의 통합 페이지 생성
    audio_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name}"
    
    # 제목 형식: [2024-05-20] 오늘의 논문 브리핑 (논문 1 | 논문 2 | ...)
    short_titles = " | ".join([t[:20] + "..." if len(t) > 20 else t for t in paper_titles_list])
    page_title = f"[{now.strftime('%Y-%m-%d')}] 통합 브리핑 ({len(valid_papers)}건)"

    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "이름": {"title": [{"text": {"content": page_title}}]},
            "날짜": {"date": {"start": now.date().isoformat()}},
            "카테고리": {"select": {"name": "Daily Update"}},
            "요약": {"rich_text": [{"text": {"content": summary_text}}]},
            "오디오": {"url": audio_url}
        }
    )
    print(f"통합 브리핑 생성 완료: {len(valid_papers)}개의 논문")

if __name__ == "__main__":
    run_bot()
