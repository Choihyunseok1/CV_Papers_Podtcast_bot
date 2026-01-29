import os
import arxiv
import openai
from notion_client import Client
import datetime
from pytz import timezone
from pydub import AudioSegment
import io
import re


NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GITHUB_USER = "Choihyunseok1"
GITHUB_REPO = "Paper2Audio"

notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


BATCH_SIZE_FULL = 2
MAX_OUT_TOKENS_SUMMARY = 3200
MAX_OUT_TOKENS_FULL_PER_BATCH = 2800

TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"
TTS_SPEED = 1.25

TTS_CHUNK_CHARS = 2000
TTS_CHUNK_OVERLAP = 0


def split_notion_text(text, max_len=1900):
    text = (text or "").strip()
    if not text:
        return []
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]


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


def build_papers_info(papers):
    papers_info = ""
    for i, p in enumerate(papers):
        papers_info += f"논문 {i+1} 제목: {p.title}\n초록: {p.summary}\n\n"
    return papers_info


def prompt_summary_and_3min(valid_papers):
    papers_info = build_papers_info(valid_papers)
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
- "시간이 없으신 분들을 위한 3분 핵심 요약입니다"로 시작할 것.
- 각 논문 제목을 말한 뒤, 논문 당 약 600자 내외로 설명할 것.
- 논문의 공식 제목은 반드시 영문으로 표기하되, 제목의 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿀 것.
- 모든 기술 약어(CNN, ViT, SOTA 등)는 100% 한글 발음으로만 표기할 것.
- 쉼표(,)를 충분히 사용해 호흡 지점을 표시할 것.
- 동료 연구자에게 설명하듯 차분한 구어체.

출력 형식:
[요약]
(내용)

[3분대본]
(내용)
""".strip()


def prompt_full_body_for_batch(batch_papers, batch_index, total_batches, start_index):
    papers_info = build_papers_info(batch_papers)

    return f"""
아래는 컴퓨터 비전 논문 배치 {batch_index}/{total_batches}입니다.
이 배치의 논문 전역 번호는 {start_index}부터 시작합니다.

{papers_info}

중요:
- 지금은 방송의 도입부와 맺음말을 쓰지 않습니다.
- "첫 번째 논문", "이번 배치", "안녕하세요", "오늘은" 같은 진행 멘트와 순서 멘트를 절대 쓰지 마세요.
- 오직 각 논문 설명 본문만 출력하세요.

분량:
- 논문 1편당 약 2200자 내외로 상세히 설명하세요.

구조:
A. 논문 제목을 한 번 말하기
B. 문제의식과 배경
C. 핵심 아이디어 한 줄 요약과 의미
D. 방법을 단계적으로 설명
E. 실험과 결과의 경향
F. 한계와 추후 과제
G. 실전 감상 포인트 2개

언어 규칙:
- 논문 제목은 영문으로, 제목의 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿀 것.
- 기술 약어(CNN, ViT, SOTA 등)는 한글 발음으로만 표기할 것.
- 쉼표(,)로 호흡, 마침표(.)로 강조.

출력 형식(반드시 준수):
TITLE: <영문 제목>
BODY:
<본문>

(논문과 논문 사이는 빈 줄 2줄)
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
    return (resp.choices[0].message.content or "").strip()


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


def sanitize_title_for_tts(title):
    if not title:
        return ""
    return re.sub(r"[:\-+/]", ",", title)


def parse_title_body_blocks(text):
    text = (text or "").strip()
    if not text:
        return []

    pattern = r"TITLE:\s*(.*?)\s*BODY:\s*(.*?)(?=(?:\n\s*TITLE:)|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    blocks = []
    for title, body in matches:
        t = title.strip()
        b = body.strip()
        if t and b:
            blocks.append((t, b))

    if blocks:
        return blocks

    fallback = []
    chunks = re.split(r"\n\s*TITLE:\s*", "\n" + text)
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if "BODY:" in c:
            t, b = c.split("BODY:", 1)
            t = t.strip()
            b = b.strip()
            if t and b:
                fallback.append((t, b))
    return fallback


def assemble_radio_script(full_batches_text, total_papers):
    intro = f"안녕하세요, 아이알씨브이 랩실의 수석 연구 비서입니다. 오늘 살펴볼 컴퓨터 비전 신규 논문은 총 {total_papers}건입니다."
    outro = "오늘의 브리핑이 여러분의 연구에 영감이 되길 바랍니다. 이상, 아이알씨브이 연구 비서였습니다. 감사합니다."

    all_blocks = []
    for batch_text in full_batches_text:
        all_blocks.extend(parse_title_body_blocks(batch_text))

    script_parts = [intro, ""]
    for i, (title, body) in enumerate(all_blocks, start=1):
        title_tts = sanitize_title_for_tts(title)
        transition = f"지금부터 {i}번째 논문입니다."
        script_parts.append(transition)
        script_parts.append(body)
        script_parts.append("")

    if len(all_blocks) < total_papers:
        script_parts.append("일부 논문 원고가 누락되어, 생성된 부분까지만 이어서 읽겠습니다.")
        script_parts.append("")

    script_parts.append(outro)
    return "\n".join(script_parts).strip()


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

    system_summary = "너는 IRCV 랩실의 수석 연구 비서이자 AI 전문 라디오 진행자야. 한국어로 요약과 3분 대본을 작성해줘."
    system_full = "너는 IRCV 랩실의 수석 연구 비서이자 AI 전문 라디오 진행자야. 한국어로 논문 본문 스크립트만 작성해줘."

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

    full_batches_text = []
    for idx, batch in enumerate(paper_batches, start=1):
        start_index = (idx - 1) * BATCH_SIZE_FULL + 1
        user_full = prompt_full_body_for_batch(batch, idx, total_batches, start_index)
        batch_text = call_gpt_text(system_full, user_full, MAX_OUT_TOKENS_FULL_PER_BATCH)
        full_batches_text.append(batch_text)

    audio_script_full = assemble_radio_script(full_batches_text, total_papers=len(valid_papers))

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

    # --------- 여기부터가 "바뀐 부분"의 핵심: 요약을 2000자 제한에 맞춰 쪼개기 ---------
    notion_children = [
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 핵심 요약"}}]}},
    ]

    for part in split_notion_text(summary_text, max_len=1900):
        notion_children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": part}}]}
        })

    notion_children += [
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 원문 링크"}}]}}
    ]
    # ---------------------------------------------------------------------

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
