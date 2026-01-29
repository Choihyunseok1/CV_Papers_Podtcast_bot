import os
import arxiv
import openai
from notion_client import Client
import datetime
from pytz import timezone
from pydub import AudioSegment
import io

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

    # 2. 시간대 설정
    seoul_tz = timezone('Asia/Seoul')
    now = datetime.datetime.now(seoul_tz)
    yesterday_6pm = (now - datetime.timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

    # 3. arXiv 논문 검색
    search = arxiv.Search(
        query="cat:cs.CV",
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    valid_papers = []
    for p in search.results():
        p_date = p.published.astimezone(seoul_tz)
        if p_date > yesterday_6pm:
            valid_papers.append(p)

    if not valid_papers:
        print("해당 시간대에 새로 올라온 논문이 없습니다.")
        return

    # 4. 프롬프트 구성 (사용자 최적화 프롬프트 유지)
    papers_info = ""
    paper_titles_list = []
    for i, p in enumerate(valid_papers):
        papers_info += f"논문 {i+1} 제목: {p.title}\n초록: {p.summary}\n\n"
        paper_titles_list.append(p.title)

    combined_prompt = f"""
    아래는 어제 저녁부터 오늘 새벽 사이에 새로 발표된 {len(valid_papers)}개의 컴퓨터 비전 논문입니다.
    
    {papers_info}

    위 논문들을 바탕으로 다음 세 가지를 작성해 주세요.

    1. [요약]
    - 노션 기록용 핵심 요약.
    - 각 논문별로 제목을 언급하고, '-함', '-임' 형태의 짧은 요약체로 2~3줄씩 작성하고, 한 줄이 끝나면 엔터로 구분할 것.
    - 각 논문 요약 시작시 '1. (논문제목)' 식으로 앞에 번호만 붙여 진행할 것
    - 논문들 사이는 줄바꿈으로 구분할 것.

    2. [전체대본] 작성 가이드라인:
    - 형식: 라디오 방송 '모닝 Computer Vision AI 브리핑' 스크립트.
    - 분량: 각 논문 제목을 말한뒤, 한 논문 당 약 500~600자 내외로 상세히 설명하여, 전체 방송이 논문당 2분정도 소요되게 할 것.
    - 구성: [도입부] - [본문: 논문] - [맺음말]의 단일 에피소드 구조.
    - 도입부: "안녕하세요, IRCV 랩실의 수석 연구 비서입니다. 오늘 살펴볼 컴퓨터 비전 신규 논문은 총 {len(valid_papers)}건입니다."로 시작할 것.
    - 호흡 조절: 
        * 문장 사이에는 충분한 쉼표(,)를 사용해 아나운서가 숨을 고르는 지점을 표시할 것.
        * 중요한 강조점 앞뒤에는 마침표(.)를 찍어 확실히 끊어 읽게 할 것.
    - 언어 처리:
        * 논문의 공식 제목은 반드시 **영문**로 표기하되, 논문 제목에 포함된 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿀 것.
        * 모든 기술 약어(CNN, ViT, SOTA 등)는 100% 한글 발음으로만 표기할 것. (예: 시엔엔, 비아이티, 소타)
    - 톤앤매너:
        * 텍스트를 읽는 것이 아니라, 동료 연구자에게 '설명'해주는 듯한 차분하고 다정한 어조.
        * "이 논문은 ~를 제안합니다" 보다는 "이 연구에서는 ~라는 흥미로운 접근을 시도했습니다" 같은 구어체 사용.
    - 마무리: "오늘의 브리핑이 여러분의 연구에 영감이 되길 바랍니다. 이상, IRCV 연구 비서였습니다. 감사합니다."
    
    3. [3분대본]: 바쁜 사람들을 위한 초압축 브리핑. 
       - 전체 논문의 핵심만 훑으며 약 3분(공백 포함 3000자 내외) 분량으로 작성.
       - "시간이 없으신 분들을 위한 3분 핵심 요약입니다"라는 멘트로 시작할 것.

    출력 형식:
    [요약]
    (통합 요약 내용)

    [전체대본]
    (통합 라디오 스크립트 내용)

    [3분대본]
    (3분대 라디오 스크립트 내용)
    """

    # 5. GPT-4o에게 통합 요청 (긴 대본 생성을 위해 max_tokens 확장)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "너는 IRCV 랩실의 수석 연구 비서이자 AI 전문 라디오 진행자야. 20분 이상의 심층 브리핑 대본을 아주 풍성하게 작성하고, 3분 압축 브리핑을 작성해줘"},
                  {"role": "user", "content": combined_prompt}],
        max_tokens=4000 
    )
    full_text = response.choices[0].message.content
    summary_text = full_text.split("[전체대본]")[0].replace("[요약]", "").strip()
    audio_script_full = full_text.split("[전체대본]")[1].split("[3분대본]")[0].strip()
    audio_script_3min = full_text.split("[3분대본]")[1].strip()
    
    # 6. 분할 TTS 및 오디오 병합 (4,000자 제한 해결)
    # 문장 단위(마침표)로 쪼개서 약 2500자씩 청크 생성
    sentences = audio_script.split('. ')
    chunks = []
    temp_chunk = ""
    for sentence in sentences:
        if len(temp_chunk) + len(sentence) < 2500:
            temp_chunk += sentence + ". "
        else:
            chunks.append(temp_chunk.strip())
            temp_chunk = sentence + ". "
    if temp_chunk:
        chunks.append(temp_chunk.strip())

    combined_audio = AudioSegment.empty()
    print(f"총 {len(chunks)}개의 파트로 나누어 음성 생성을 시작합니다...")

    for i, chunk in enumerate(chunks):
        audio_part_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="onyx",
            input=chunk,
            speed=1.2
        )
        # 메모리에서 직접 오디오 데이터 로드
        part_stream = io.BytesIO(audio_part_response.content)
        audio_segment = AudioSegment.from_file(part_stream, format="mp3")
        combined_audio += audio_segment
        print(f"파트 {i+1}/{len(chunks)} 생성 완료")

    # [전체대본]최종 파일 저장
    today_date = now.strftime('%Y%m%d') 
    file_name = f"CV_Daily_Briefing_{today_date}.mp3" 
    full_file_path = os.path.join(audio_dir, file_name)
    combined_audio.export(full_file_path, format="mp3")

    #6-2 [3분대본]
    file_name_3min = f"3Min_Summary_{today_date}.mp3"
    full_file_path_3min = os.path.join(audio_dir, file_name_3min)
    
    response_3min = client.audio.speech.create(
        model="tts-1-hd", # 3분용은 가볍게 기본 모델 사용 가능
        voice="onyx",
        input=audio_script_3min,
        speed=1.2
    )
    response_3min.stream_to_file(full_file_path_3min)

    # 7. 노션에 페이지 생성 (기존 형식 유지)
    audio_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name}"
    audio_url_3min = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name_3min}"
    page_title = f"[{now.strftime('%Y-%m-%d')}] 통합 브리핑 ({len(valid_papers)}건)"

    notion_children = [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": "📄 논문 핵심 요약"}}]}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": summary_text}}]}},
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": "🔗 논문 원문 링크"}}]}}
    ]

    for i, p in enumerate(valid_papers):
        notion_children.append({
            "object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {"type": "text", "text": {"content": f"{i+1}. {p.title} "}},
                    {"type": "text", "text": {"content": "[PDF]", "link": {"url": p.pdf_url}}, "annotations": {"bold": True, "color": "blue"}}
                ]
            }
        })

    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "이름": {"title": [{"text": {"content": page_title}}]},
            "날짜": {"date": {"start": now.date().isoformat()}},
            "오디오": {"url": audio_url},
            "3분 논문": {"url": audio_url_3min}
        },
        children=notion_children
    )
    print(f"통합 브리핑 생성 완료: {len(valid_papers)}개의 논문")

if __name__ == "__main__":
    run_bot()
