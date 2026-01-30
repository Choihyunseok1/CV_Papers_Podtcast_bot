import os
import re
import io
import datetime
from urllib.parse import quote

import arxiv
import openai
import requests
from notion_client import Client
from pytz import timezone
from pydub import AudioSegment


NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

GITHUB_USER = "Choihyunseok1"
GITHUB_REPO = "CV_Papers_Podtcast_bot"

notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Top10만 생성
TOP_K = 10
ARXIV_MAX_RESULTS = 500

# 본문 생성 배치 크기
BATCH_SIZE_FULL = 2

# OpenAI 출력 길이
MAX_OUT_TOKENS_SUMMARY = 4000
MAX_OUT_TOKENS_FULL_PER_BATCH = 2800

# TTS
TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"
TTS_SPEED = 1.25
TTS_CHUNK_CHARS = 2000
TTS_CHUNK_OVERLAP = 0

# arXiv announce 기준 시각 (Eastern Time 20:00)
ARXIV_ANNOUNCE_HOUR_ET = 20
ARXIV_ANNOUNCE_MINUTE_ET = 0


S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = ",".join([
    "title",
    "authors.name",
    "authors.hIndex",
    "authors.paperCount",
    "authors.citationCount",
    "externalIds",
    "url",
    "openAccessPdf",
    "citationCount",
    "venue",
    "year",
])


def s2_headers():
    headers = {"Content-Type": "application/json"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


def chunk_list(xs, n):
    out = []
    for i in range(0, len(xs), n):
        out.append(xs[i:i + n])
    return out


def arxiv_id_from_entry_id(entry_id: str) -> str:
    if not entry_id:
        return ""
    return entry_id.rstrip("/").split("/")[-1].strip()


def fetch_s2_papers_batch(arxiv_ids):
    results = {}
    ids = [f"ARXIV:{aid}" for aid in arxiv_ids if aid]
    if not ids:
        return results

    url = f"{S2_API_BASE}/paper/batch?fields={quote(S2_FIELDS)}"
    body = {"ids": ids}

    try:
        resp = requests.post(url, headers=s2_headers(), json=body, timeout=30)
    except Exception as e:
        print("S2 request error:", str(e))
        return results

    if resp.status_code != 200:
        print("S2 batch fetch failed:", resp.status_code, (resp.text or "")[:300])
        return results

    data = resp.json() or []
    for item in data:
        if not item:
            continue
        ext = (item.get("externalIds") or {})
        arxiv_id = (ext.get("ArXiv") or "").strip()
        if arxiv_id:
            results[arxiv_id] = item
    return results


def get_last_announce_window_et(now_et):
    # now_et이 20:00 ET 이후면: window_start = 오늘 20:00
    # now_et이 20:00 ET 이전이면: window_start = 어제 20:00
    today_announce = now_et.replace(
        hour=ARXIV_ANNOUNCE_HOUR_ET,
        minute=ARXIV_ANNOUNCE_MINUTE_ET,
        second=0,
        microsecond=0
    )
    if now_et >= today_announce:
        window_start = today_announce
    else:
        window_start = today_announce - datetime.timedelta(days=1)

    window_end = window_start + datetime.timedelta(days=1)
    return window_start, window_end


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
아래는 오늘 arXiv에 새로 공개된 {len(valid_papers)}개의 컴퓨터 비전 논문입니다.

{papers_info}

위 논문들을 바탕으로 다음 두 가지를 작성해 주세요.

1) [요약]
- 노션 기록용 핵심 요약입니다.
- 각 논문별로 제목을 언급하고, '-함', '-임' 형태의 짧은 요약체로 3줄씩 작성해 주세요.
- 한 줄이 끝나면 반드시 엔터로 구분해 주세요.
- 각 논문 요약 시작은 '1. (논문제목)' 형식으로 번호만 붙여 주세요.
- 논문들 사이는 줄바꿈으로 구분해 주세요.

2) [3분대본]
- "시간이 없으신 분들을 위한 3분 핵심 요약입니다"로 시작해 주세요.
- 모든 논문을 빠짐없이 포함해 주세요.
- 각 논문 제목을 말한 뒤, 논문 당 약 350~450자 내외로 설명해 주세요.
- 전체 길이는 약 3분(±15초) 분량이 되도록 조절해 주세요.
- 논문 수가 많을 경우 각 논문의 설명 길이를 자동으로 줄여 전체 분량을 유지해 주세요.
- 논문 제목은 반드시 영문으로 표기하되, 제목의 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿔 주세요.
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용해 주세요.
- diffusion, transformer, attention, encoder, decoder, latent, alignment, distillation, benchmark, dataset 같은 전문 용어는 번역하지 말고 영어 그대로 사용해 주세요.
- 쉼표(,)를 충분히 사용해 호흡 지점을 표시해 주세요.
- 동료 연구자에게 설명하듯 차분한 구어체로 쓰되, 반드시 공적인 라디오 톤의 존댓말로 작성해 주세요.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠)는 사용하지 말아 주세요.

[마무리 규칙]
- 모든 논문 설명이 끝난 뒤, 아래 톤의 아웃트로 멘트를 한 문단으로 추가해 주세요.
- 감사 인사나 일상적인 인삿말은 사용하지 말아 주세요.
- 더 자세한 내용이 전체 브리핑에 있다는 점을 자연스럽게 안내해 주세요.

아웃트로 예시 톤:
"보다 자세한 내용은 전체 브리핑에서 이어서 다룹니다.
지금까지 오늘의 컴퓨터 비전 논문 3분 핵심 요약이었습니다."

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
- 논문 1편당 약 1800~2300자 내외로 상세히 설명하세요.

언어 규칙:
- 쉼표(,)로 호흡, 마침표(.)로 강조.
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용할 것.
- diffusion, transformer, attention, encoder, decoder, latent 등 전문 용어는 번역하지 말고 영어 그대로 사용할 것.
- 동료 연구자에게 설명하듯 차분한 구어체.
- 오디오 스크립트에서 구조/순서를 직접적으로 말하지 말 것(예: "A.", "B.", "첫째", "다음으로", "이어서").
- 전체 브리핑은 반드시 공적인 라디오 톤의 존댓말로 작성할 것.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠)는 사용하지 말 것.

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
    intro = f"안녕하세요, 아이알씨브이 랩실의 수석 연구 비서입니다. 오늘 선별된 컴퓨터 비전 신규 논문은 총 {total_papers}건입니다."
    outro = "이상으로 오늘의 브리핑을 마치겠습니다."

    all_blocks = []
    for batch_text in full_batches_text:
        all_blocks.extend(parse_title_body_blocks(batch_text))

    script_parts = [intro, ""]
    for i, (title, body) in enumerate(all_blocks, start=1):
        title_tts = sanitize_title_for_tts(title)

        if i == 1:
            transition = "첫 번째로 살펴볼 논문입니다."
        else:
            transition = f"{i}번째 논문입니다."

        script_parts.append(transition)
        script_parts.append(f"논문 제목은 {title_tts} 입니다.")
        script_parts.append(body)
        script_parts.append("")

    if len(all_blocks) < total_papers:
        script_parts.append("일부 원고 생성이 누락되어, 생성된 부분까지만 제공됩니다.")
        script_parts.append("")

    script_parts.append(outro)
    return "\n".join(script_parts).strip()


def compute_author_score_0_100(s2_paper):
    # Author score 비중을 올리기 위해, 0~100으로 강하게 스케일
    authors = (s2_paper.get("authors") or [])
    if not authors:
        return 5

    def one_author(a):
        h = a.get("hIndex") or 0
        pc = a.get("paperCount") or 0
        cc = a.get("citationCount") or 0

        score = 0

        # h-index
        if h >= 60:
            score += 55
        elif h >= 40:
            score += 45
        elif h >= 25:
            score += 35
        elif h >= 15:
            score += 25
        elif h >= 8:
            score += 15
        else:
            score += 8

        # 생산성
        if pc >= 150:
            score += 20
        elif pc >= 80:
            score += 15
        elif pc >= 40:
            score += 10
        elif pc >= 15:
            score += 6
        else:
            score += 3

        # 누적 인용(거칠게)
        if cc >= 20000:
            score += 25
        elif cc >= 8000:
            score += 18
        elif cc >= 2000:
            score += 12
        elif cc >= 300:
            score += 7
        else:
            score += 3

        return min(100, score)

    scores = sorted([one_author(a) for a in authors if a], reverse=True)
    if not scores:
        return 5

    if len(scores) == 1:
        base = scores[0]
    else:
        base = (scores[0] * 0.7) + (scores[1] * 0.3)

    return int(round(max(0, min(100, base))))


def compute_signal_score_0_100(arxiv_paper):
    # 무거운 기준 제거: 초록 텍스트 기반의 가벼운 신호만 사용
    abs_ = (arxiv_paper.summary or "").lower()

    good = [
        "benchmark", "dataset", "ablation", "analysis", "failure", "limitation",
        "code", "github", "open-source", "open source", "reproduc",
        "vision-language", "multimodal", "distillation", "foundation model",
        "detection", "segmentation", "tracking", "3d", "depth", "pose", "video"
    ]

    hits = 0
    for k in good:
        if k in abs_:
            hits += 1

    # 0~100으로는 과하니 완만하게
    if hits >= 10:
        score = 80
    elif hits >= 7:
        score = 65
    elif hits >= 4:
        score = 50
    elif hits >= 2:
        score = 35
    elif hits >= 1:
        score = 25
    else:
        score = 15

    return score


def compute_penalty_0_30(arxiv_paper, s2_paper):
    # 과장/근거부족만 가볍게 감점
    abs_ = (arxiv_paper.summary or "").lower()
    penalty = 0

    hype = ["revolutionary", "breakthrough", "novel paradigm", "unprecedented", "game-changing"]
    if any(k in abs_ for k in hype):
        penalty += 8

    if ("experiment" not in abs_) and ("evaluation" not in abs_) and ("benchmark" not in abs_) and ("dataset" not in abs_):
        penalty += 10

    authors = (s2_paper.get("authors") or [])
    if authors:
        hs = [(a.get("hIndex") or 0) for a in authors if a]
        if hs and max(hs) < 5:
            penalty += 12

    return max(0, min(30, penalty))


def total_score_0_100(arxiv_paper, s2_paper):
    # Author score 비중 상승 (80%), signal (20%), penalty 차감
    a = compute_author_score_0_100(s2_paper)
    s = compute_signal_score_0_100(arxiv_paper)
    p = compute_penalty_0_30(arxiv_paper, s2_paper)

    score = (a * 0.80) + (s * 0.20) - p
    return int(round(max(0, min(100, score))))


def select_top_k_papers(valid_papers, k):
    # arXiv id 수집
    arxiv_ids = []
    paper_by_id = {}

    for p in valid_papers:
        aid = arxiv_id_from_entry_id(getattr(p, "entry_id", "") or "")
        if not aid:
            aid = arxiv_id_from_entry_id(getattr(p, "pdf_url", "") or "")
        if aid:
            arxiv_ids.append(aid)
            paper_by_id[aid] = p

    # S2 batch 호출
    s2_map = {}
    for chunk in chunk_list(arxiv_ids, 200):
        s2_map.update(fetch_s2_papers_batch(chunk))

    scored = []
    for aid in arxiv_ids:
        p = paper_by_id.get(aid)
        s2 = s2_map.get(aid)

        if not p:
            continue

        if not s2:
            # S2 누락 시 낮은 점수
            scored.append((10, aid))
            continue

        sc = total_score_0_100(p, s2)
        scored.append((sc, aid))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:k]
    top_ids = [aid for _, aid in top]

    selected = [paper_by_id[aid] for aid in top_ids if aid in paper_by_id]
    return selected, scored


def run_bot():
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    seoul_tz = timezone("Asia/Seoul")
    et_tz = timezone("America/New_York")

    now_kst = datetime.datetime.now(seoul_tz)
    now_et = now_kst.astimezone(et_tz)

    window_start_et, window_end_et = get_last_announce_window_et(now_et)

    search = arxiv.Search(
        query="cat:cs.CV",
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    candidates = []
    for p in search.results():
        p_et = p.published.astimezone(et_tz)
        if window_start_et <= p_et < window_end_et:
            candidates.append(p)

    if not candidates:
        print("해당 announce window에서 새로 올라온 논문이 없습니다.")
        return

    top_papers, scored_all = select_top_k_papers(candidates, TOP_K)

    if not top_papers:
        print("Top 논문을 선택하지 못했습니다(S2 누락 등).")
        return

    system_summary = "당신은 연구실의 수석 연구 비서이자 AI 전문 라디오 진행자입니다. 한국어로 요약과 3분 대본을 작성해 주세요. 존댓말을 유지해 주세요."
    system_full = "당신은 연구실의 수석 연구 비서이자 AI 전문 라디오 진행자입니다. 한국어로 논문 본문 스크립트만 작성해 주세요. 존댓말을 유지해 주세요."

    user_summary = prompt_summary_and_3min(top_papers)
    summary_out = call_gpt_text(system_summary, user_summary, MAX_OUT_TOKENS_SUMMARY)

    if "[3분대본]" in summary_out:
        summary_text = summary_out.split("[3분대본]")[0].replace("[요약]", "").strip()
        audio_script_3min = summary_out.split("[3분대본]")[1].strip()
    else:
        summary_text = summary_out.replace("[요약]", "").strip()
        audio_script_3min = ""

    paper_batches = [top_papers[i:i + BATCH_SIZE_FULL] for i in range(0, len(top_papers), BATCH_SIZE_FULL)]
    total_batches = len(paper_batches)

    full_batches_text = []
    for idx, batch in enumerate(paper_batches, start=1):
        start_index = (idx - 1) * BATCH_SIZE_FULL + 1
        user_full = prompt_full_body_for_batch(batch, idx, total_batches, start_index)
        batch_text = call_gpt_text(system_full, user_full, MAX_OUT_TOKENS_FULL_PER_BATCH)
        full_batches_text.append(batch_text)

    audio_script_full = assemble_radio_script(full_batches_text, total_papers=len(top_papers))

    combined_audio = synthesize_tts_to_audio(
        audio_script_full,
        tts_chunk_chars=TTS_CHUNK_CHARS,
        overlap=TTS_CHUNK_OVERLAP
    )

    today_date = now_kst.strftime("%Y%m%d")
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

    page_title = f"{now_kst.strftime('%Y-%m-%d')} 모닝 브리핑 (Top {len(top_papers)})"

    notion_children = [
        {"object": "block", "type": "callout",
         "callout": {"rich_text": [{"type": "text", "text": {"content": f"arXiv announce window 기준 후보 {len(candidates)}개 중, Author 중심 점수로 상위 {len(top_papers)}개만 브리핑합니다."}}],
                    "icon": {"emoji": "🧭"}}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 핵심 요약"}}]}}
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
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 원문 링크 (Top 10)"}}]}}
    ]

    for i, p in enumerate(top_papers):
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
            "요약 & 논문링크": {"title": [{"text": {"content": page_title}}]},
            "날짜": {"date": {"start": now_kst.date().isoformat()}},
            "전체 브리핑": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "▶ 전체 브리핑 다운",
                            "link": {"url": audio_url}
                        }
                    }
                ]
            },
            "3분 요약": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "▶ 3분 요약 다운",
                            "link": {"url": audio_url_3min}
                        }
                    }
                ]
            }
        },
        children=notion_children
    )

    print(f"완료: 후보 {len(candidates)}개 중 Top {len(top_papers)}개 생성")


if __name__ == "__main__":
    run_bot()
