import os
import arxiv
import openai
from notion_client import Client
import datetime
from pytz import timezone
from pydub import AudioSegment
import io


NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GITHUB_USER = "Choihyunseok1"
GITHUB_REPO = "Paper2Audio"

notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


BATCH_SIZE_FULL = 2
MAX_OUT_TOKENS_SUMMARY = 2500
MAX_OUT_TOKENS_FULL_PER_BATCH = 3900

TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"
TTS_SPEED = 1.2

TTS_CHUNK_CHARS = 2000
TTS_CHUNK_OVERLAP = 0


def chunk_text_by_chars(text, chunk_chars=2000, overlap=0):
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    step = max(1, chunk_chars - overlap)
    while i < n:
        chunk = text[i:i + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def build_papers_info(valid_papers):
    papers_info = ""
    titles = []
    for i, p in enumerate(valid_papers):
        papers_info += f"논문 {i+1} 제목: {p.title}\n초록: {p.summary}\n\n"
        titles.append(p.title)
    return papers_info, titles


def prompt_summary_and_3min(valid_papers):
    papers_info, _ = build_papers_info(valid_papers)

    return f"""
아래는 어제 저녁부터 오늘 새벽 사이에 새로 발표된 {len(valid_papers)}개의 컴퓨터 비전 논문입니다.

{papers_info}

위 논문들을 바탕으로 다음 두 가지를 작성해 주세요.

1. [요약]
- 노션 기록용 핵심 요약.
- 각 논문별로 제목을 언급하고, '-함', '-임' 형태의 짧은 요약체로 2~3줄씩 작성할 것
- 한 줄이 끝나면 반드시 엔터로 구분해서 보기 편하게 만들 것
- 각 논문 요약 시작시 '1. (논문제목)' 식으로 앞에 번호만 붙여 진행할 것
- 논문들 사이는 줄바꿈으로 구분할 것.

2. [3분대본]
- 바쁜 사람들을 위한 초압축 브리핑.
- "시간이 없으신 분들을 위한 3분 핵심 요약입니다"라는 멘트로 시작할 것.
- 각 논문 제목을 말한 뒤, 논문 당 약 600자 내외로 설명할 것.
- 논문의 공식 제목은 반드시 영문으로 표기하되, 제목에 포함된 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿀 것.
- 모든 기술 약어(CNN, ViT, SOTA 등)는 100% 한글 발음으로만 표기할 것.
- 문장 사이에는 충분한 쉼표(,)를 사용해 호흡 지점을 표시할 것.
- 텍스트를 읽는 것이 아니라, 동료 연구자에게 설명해주는 듯한 차분하고 다정한 어조.

출력 형식:
[요약]
(내용)

[3분대본]
(내용)
""".strip()


def prompt_full_body_for_batch(batch_papers, batch_index, total_batches):
    papers_info, _ = build_papers_info(batch_papers)

    return f"""
아래는 컴퓨터 비전 논문 배치 {batch_index}/{total_batches}입니다.

{papers_info}

너의 목표는 오디오용 방송 원고를 만드는 것입니다.
중요: 지금은 방송의 도입부와 맺음말을 쓰지 않습니다. 오직 각 논문의 본문만 씁니다.

[작성 규칙]
- 형식: 라디오 방송 본문 스크립트.
- 분량: 논문 1편당 약 4000자 내외로 상세히 설명할 것. 배치에 포함된 논문 수만큼 각각 작성할 것.
- 각 논문은 아래 구조를 반드시 따를 것:
  A. 논문 제목을 한 번 말하기
  B. 문제의식과 배경
  C. 핵심 아이디어 한 줄 요약 후, 왜 중요한지
  D. 방법: 구성 요소를 단계적으로 설명
  E. 실험/결과: 무엇을 비교했고 어떤 경향이 나왔는지
  F. 한계/추후 과제: 조심스럽게 언급
  G. 실전 감상: 연구자가 얻을 수 있는 포인트 2~3개

[언어 규칙]
- 논문의 공식 제목은 반드시 영문으로 표기하되, 제목에 포함된 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿀 것.
- 모든 기술 약어(CNN, ViT, SOTA 등)는 100% 한글 발음으로만 표기할 것.
- 호흡 조절을 위해 쉼표(,)를 충분히 사용하고, 강조 지점은 마침표(.)로 끊어 읽게 할 것.
- "제안합니다" 같은 딱딱한 톤보다, "이 연구에서는 이런 접근을 시도했습니다" 같은 구어체를 사용할 것.

[출력 형식]
- 배치에 포함된 논문을 1번부터 순서대로 작성.
- 각 논문 시작은 반드시 다음처럼:
  (1) TITLE: <영문 제목>
  BODY:
  <본문>

- 논문과 논문 사이는 빈 줄 2줄로 구분.
""".strip()


def call_gpt_text(system_text, user_text, max_tokens):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


def synthesize_tts_to_audio(text, tts_chunk_chars=2000, overlap=0):
    chunks = chunk_text_by_chars(text, chunk_chars=tts_chunk_chars, overlap=overlap)
    combined = AudioSegment.empty()

    for chunk in chunks:
        audio_part_response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=chunk,
            speed=TTS_SPEED
        )
        part_stream = io.BytesIO(audio_part_response.content)
        segment = AudioSegment.from_file(part_stream, format="mp3")
        combined += segment

    return combined


def run_bot():
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    seoul_tz = timezone("Asia/Seoul")
    now = datetime.datetime.now(seoul_tz)
    yesterday_6pm = (now - datetime.timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

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

    system_summary = "너는 IRCV 랩실의 수석 연구 비서이자 AI 전문 라디오 진행자야. 한국어로 자연스럽고 정확하게, 요약과 3분 대본을 작성해줘."
    system_full = "너는 IRCV 랩실의 수석 연구 비서이자 AI 전문 라디오 진행자야. 한국어로 자연스럽고 자세하게, 논문 본문 스크립트를 작성해줘."

    user_summary = prompt_summary_and_3min(valid_papers)
    summary_out = call_gpt_text(system_summary, user_summary, MAX_OUT_TOKENS_SUMMARY)

    if "[3분대본]" in summary_out:
        summary_text = summary_out.split("[3분대본]")[0].replace("[요약]", "").strip()
        audio_script_3min = summary_out.split("[3분대본]")[1].strip()
    else:
        summary_text = summary_out.replace("[요약]", "").strip()
        audio_script_3min = ""

    paper_batches = [valid_papers[i:i + BATCH_SIZE_FULL] for i in range(0, len(valid_papers), BATCH_SIZE_FULL)]
    total_batches = len(paper_batches)

    full_bodies = []
    for idx, batch in enumerate(paper_batches, start=1):
        user_full = prompt_full_body_for_batch(batch, idx, total_batches)
        batch_body = call_gpt_text(system_full, user_full, MAX_OUT_TOKENS_FULL_PER_BATCH)
        full_bodies.append(batch_body)

    intro = f"안녕하세요, 아이아르시브이 랩실의 수석 연구 비서입니다. 오늘 살펴볼 컴퓨터 비전 신규 논문은 총 {len(valid_papers)}건입니다."
    outro = "오늘의 브리핑이 여러분의 연구에 영감이 되길 바랍니다. 이상, 아이아르시브이 연구 비서였습니다. 감사합니다."
    audio_script_full = intro + "\n\n" + "\n\n".join(full_bodies) + "\n\n" + outro

    combined_audio = synthesize_tts_to_audio(
        audio_script_full,
        tts_chunk_chars=TTS_CHUNK_CHARS,
        overlap=TTS_CHUNK_OVERLAP
    )

    today_date = now.strftime("%Y%m%d")
    file_name_full = f"CV_Daily_Briefing_{today_date}.mp3"
    full_file_path = os.path.join(audio_dir, file_name_full)
    combined_audio.export(full_file_path, format="mp3")

    file_name_3min = f"3Min_Summary_{today_date}.mp3"
    full_file_path_3min = os.path.join(audio_dir, file_name_3min)

    if audio_script_3min.strip():
        audio_3min = synthesize_tts_to_audio(
            audio_script_3min,
            tts_chunk_chars=TTS_CHUNK_CHARS,
            overlap=TTS_CHUNK_OVERLAP
        )
        audio_3min.export(full_file_path_3min, format="mp3")
    else:
        open(full_file_path_3min, "wb").close()

    audio_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name_full}"
    audio_url_3min = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name_3min}"
    page_title = f"[{now.strftime('%Y-%m-%d')}] 통합 브리핑 ({len(valid_papers)}건)"

    notion_children = [
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 핵심 요약"}}]}},
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": [{"type": "text", "text": {"content": summary_text}}]}},
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 원문 링크"}}]}}
    ]

    for i, p in enumerate(valid_papers):
        notion_children.append({
            "object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {"type": "text", "text": {"content": f"{i + 1}. {p.title} "}},
                    {"type": "text", "text": {"content": "PDF", "link": {"url": p.pdf_url}},
                     "annotations": {"bold": True, "color": "blue"}}
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
