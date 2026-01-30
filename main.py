import os
import re
import io
import math
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import arxiv
import openai
import requests
from notion_client import Client as NotionClient
from zoneinfo import ZoneInfo
from pydub import AudioSegment


# =========================
# ENV
# =========================
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "").strip()
DATABASE_ID = os.environ.get("DATABASE_ID", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

GITHUB_USER = os.environ.get("GITHUB_USER", "Choihyunseok1")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "CV_Papers_Podtcast_bot")

if not NOTION_TOKEN or not DATABASE_ID or not OPENAI_API_KEY:
    raise RuntimeError("Missing env: NOTION_TOKEN / DATABASE_ID / OPENAI_API_KEY")

notion = NotionClient(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# =========================
# CONFIG
# =========================
ARXIV_CATEGORY_QUERY = "cat:cs.CV"
ARXIV_MAX_RESULTS = 800

TOPK_SAVE = 10
PRESELECT_K = 60

BATCH_SIZE_FULL = 2

MAX_OUT_TOKENS_SUMMARY = 4000
MAX_OUT_TOKENS_FULL_PER_BATCH = 2800

TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"
TTS_SPEED = 1.25

TTS_CHUNK_CHARS = 2000
TTS_CHUNK_OVERLAP = 0

SEOUL_TZ = ZoneInfo("Asia/Seoul")
ET_TZ = ZoneInfo("America/New_York")

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_TIMEOUT = 10


# NEGATIVE_HYPE = ["revolutionary", "breakthrough", "sota", "state-of-the-art"]
CODE_HINTS = ["github.com", "gitlab.com", "bitbucket.org", "project page", "code"]


@dataclass
class ScoredPaper:
    paper: Any
    arxiv_id: str
    score: float
    score_detail: Dict[str, float]


# =========================
# TIME WINDOW: arXiv announce 기준
# (ET 14:00 cutoff / ET 20:00 announce)
# - Tue/Wed/Thu: (prev day 14:00) ~ (today 14:00)
# - Mon: (Fri 14:00) ~ (Mon 14:00)
# - Sun: (Thu 14:00) ~ (Fri 14:00)
# =========================
def is_announce_day_et(d: datetime.date) -> bool:
    return d.weekday() in (0, 1, 2, 3, 6)


def latest_announce_date_et(now_et: datetime.datetime) -> datetime.date:
    d = now_et.date()
    if (not is_announce_day_et(d)) or (now_et.time() < datetime.time(20, 0)):
        d = d - datetime.timedelta(days=1)
        while not is_announce_day_et(d):
            d = d - datetime.timedelta(days=1)
        return d
    return d


def compute_announce_window_et(now_et: datetime.datetime) -> Tuple[datetime.datetime, datetime.datetime, datetime.date]:
    ad = latest_announce_date_et(now_et)

    if ad.weekday() == 6:
        end_date = ad - datetime.timedelta(days=2)
        end_et = datetime.datetime(end_date.year, end_date.month, end_date.day, 14, 0, tzinfo=ET_TZ)
        start_et = end_et - datetime.timedelta(days=1)
        return start_et, end_et, ad

    if ad.weekday() == 0:
        end_et = datetime.datetime(ad.year, ad.month, ad.day, 14, 0, tzinfo=ET_TZ)
        start_et = end_et - datetime.timedelta(days=3)
        return start_et, end_et, ad

    end_et = datetime.datetime(ad.year, ad.month, ad.day, 14, 0, tzinfo=ET_TZ)
    start_et = end_et - datetime.timedelta(days=1)
    return start_et, end_et, ad


# =========================
# UTIL
# =========================
def split_notion_text(text: str, max_len: int = 1900) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    return [t[i:i + max_len] for i in range(0, len(t), max_len)]


def number_to_korean_ordinal(n: int) -> str:
    mapping = {
        1: "첫 번째",
        2: "두 번째",
        3: "세 번째",
        4: "네 번째",
        5: "다섯 번째",
        6: "여섯 번째",
        7: "일곱 번째",
        8: "여덟 번째",
        9: "아홉 번째",
        10: "열 번째",
    }
    return mapping.get(n, f"{n}번째")


def chunk_text_by_chars(text: str, chunk_chars: int = 2000, overlap: int = 0) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    chunks = []
    i = 0
    n = len(t)
    step = max(1, chunk_chars - overlap)
    while i < n:
        chunk = t[i:i + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def sanitize_title_for_tts(title: str) -> str:
    if not title:
        return ""
    return re.sub(r"[:\-+/]", ",", title)


def extract_arxiv_id(entry_id: str) -> str:
    if not entry_id:
        return ""
    m = re.search(r"arxiv\.org/abs/([^v]+)(?:v\d+)?", entry_id)
    if m:
        return m.group(1).strip()
    return entry_id.rsplit("/", 1)[-1].replace("v1", "").replace("v2", "").strip()


def safe_lower(s: str) -> str:
    return (s or "").lower()


def format_kst_date(now_kst: datetime.datetime) -> str:
    return f"{now_kst.month}월 {now_kst.day}일"


def format_kst_date_iso(now_kst: datetime.datetime) -> str:
    return now_kst.strftime("%Y-%m-%d")


# =========================
# ARXIV FETCH
# =========================
def fetch_arxiv_candidates_in_window(start_et: datetime.datetime, end_et: datetime.datetime) -> List[Any]:
    aclient = arxiv.Client(page_size=200, delay_seconds=0.0, num_retries=3)

    search = arxiv.Search(
        query=ARXIV_CATEGORY_QUERY,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    candidates = []
    for p in aclient.results(search):
        pub_et = p.published.astimezone(ET_TZ)
        if pub_et < start_et:
            break
        if start_et <= pub_et < end_et:
            candidates.append(p)

    return candidates


# =========================
# STAGE 0: LIGHT PRE-SCORE
# =========================
def prescore_text(p: Any) -> float:
    title = safe_lower(p.title)
    abst = safe_lower(p.summary)
    text = title + " " + abst

    code_hit = any(h in text for h in CODE_HINTS)

    score = 0.0
    score += 6.0 if code_hit else 0.0
    score -= min(hype_hits * 2.0, 6.0)
    return score


# =========================
# SEMANTIC SCHOLAR
# =========================
def s2_headers() -> Dict[str, str]:
    h = {"User-Agent": "cv-papers-podcast-bot/1.0"}
    if SEMANTIC_SCHOLAR_API_KEY:
        h["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return h


def fetch_s2_paper_by_arxiv(arxiv_id: str) -> Optional[Dict[str, Any]]:
    if not arxiv_id:
        return None
    url = f"{S2_BASE}/paper/ARXIV:{arxiv_id}"
    fields = "title,authors.name,authors.hIndex,authors.paperCount,externalIds,url"
    try:
        r = requests.get(url, headers=s2_headers(), params={"fields": fields}, timeout=S2_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def compute_author_score(s2: Optional[Dict[str, Any]]) -> float:
    if not s2 or "authors" not in s2:
        return 10.0

    authors = s2.get("authors") or []
    if not authors:
        return 10.0

    h_list = []
    pc_list = []
    for a in authors:
        h = a.get("hIndex")
        pc = a.get("paperCount")
        if isinstance(h, int):
            h_list.append(h)
        if isinstance(pc, int):
            pc_list.append(pc)

    if not h_list:
        h_list = [0]

    h_list_sorted = sorted(h_list, reverse=True)
    top2 = h_list_sorted[:2]
    h_avg = sum(top2) / max(1, len(top2))

    if h_avg >= 40:
        base = 62.0
    elif h_avg >= 25:
        base = 52.0
    elif h_avg >= 15:
        base = 40.0
    elif h_avg >= 8:
        base = 28.0
    else:
        base = 18.0

    pc_max = max(pc_list) if pc_list else 0
    if pc_max >= 200:
        base += 6.0
    elif pc_max >= 100:
        base += 4.0
    elif pc_max >= 50:
        base += 2.0

    return max(0.0, min(70.0, base))


def compute_content_score(p: Any) -> float:
    title = safe_lower(p.title)
    abst = safe_lower(p.summary)
    text = title + " " + abst

    code_hit = any(h in text for h in CODE_HINTS)

    return 10.0 if code_hit else 0.0



def compute_penalty(p: Any) -> float:
    title = safe_lower(p.title)
    abst = safe_lower(p.summary)
    text = title + " " + abst

    hype_hits = sum(1 for k in NEGATIVE_HYPE if k in text)
    if hype_hits == 0:
        return 0.0
    return -min(10.0, hype_hits * 3.0)


def score_paper(p: Any) -> ScoredPaper:
    arxiv_id = extract_arxiv_id(getattr(p, "entry_id", ""))
    s2 = fetch_s2_paper_by_arxiv(arxiv_id)

    author = compute_author_score(s2)
    content = compute_content_score(p)
    penalty = 0.0
    
    total = author + content + penalty
    total = max(0.0, min(100.0, total * (100.0 / 80.0)))

    return ScoredPaper(
        paper=p,
        arxiv_id=arxiv_id,
        score=total,
        score_detail={"author": author, "content": content, "penalty": penalty}
    )


# =========================
# GPT PROMPTS
# =========================
def build_papers_info(papers: List[Any]) -> str:
    out = []
    for i, p in enumerate(papers, start=1):
        out.append(f"논문 {i} 제목: {p.title}\n초록: {p.summary}\n")
    return "\n".join(out).strip()


def prompt_summary_and_3min(top_papers: List[Any], kst_date_iso: str, kst_date_korean: str) -> str:
    papers_info = build_papers_info(top_papers)
    return f"""
오늘은 {kst_date_korean}이며, 이 날짜 기준으로 서술하십시오.
아래는 arXiv cs.CV에서 {kst_date_iso}자 기준으로 선별된 상위 {len(top_papers)}편 논문입니다.
(중요: 아래에 포함된 논문만 요약 및 오디오 대본을 작성하십시오. 나머지 논문은 절대 다루지 마십시오.)

{papers_info}

위 논문들을 바탕으로 다음 두 가지를 작성해 주세요.

1. [요약]
- 노션 기록용 핵심 요약입니다.
- 각 논문별로 제목을 언급하고, '-함', '-임' 형태의 짧은 요약체로 3줄씩 작성하십시오.
- 한 줄이 끝나면 반드시 엔터로 구분하십시오.
- 각 논문 요약 시작 시 '1. (논문제목)' 형식으로 번호를 붙이십시오.
- 논문들 사이는 줄바꿈으로 구분하십시오.

2. [3분대본]
- "시간이 없으신 분들을 위한 3분 핵심 요약입니다"로 시작하십시오.
- 위에 제공된 상위 {len(top_papers)}편 논문을 빠짐없이 포함하십시오.
- 각 논문 제목을 말한 뒤, 논문 당 길이를 자동 조절하여 전체가 약 3분(±15초)이 되도록 구성하십시오.
- 논문의 공식 제목은 반드시 영문으로 표기하되, 제목의 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꾸십시오.
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용하십시오.
- 전문 기술 용어는 번역하지 말고 반드시 영어 원어 그대로 사용하십시오.
- 쉼표(,)를 충분히 사용하여 호흡 지점을 표시하십시오.
- 전체 브리핑은 공적인 라디오 방송 톤의 존댓말로 작성하십시오.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠)는 사용하지 마십시오.
- 연구 비서가 공식적으로 설명하는 말투를 유지하십시오.

마무리 규칙
- 모든 논문 설명이 끝난 뒤, 아래 톤의 아웃트로 멘트를 한 문단으로 추가하십시오.
- 감사 인사나 일상적인 인삿말은 사용하지 마십시오.
- 더 자세한 내용이 전체 브리핑에 있다는 점을 자연스럽게 안내하십시오.

아웃트로 톤 예시
"보다 자세한 내용은 전체 브리핑에서 이어서 다룹니다.
지금까지 오늘의 컴퓨터 비전 논문 3분 핵심 요약이었습니다."

출력 형식
[요약]
(내용)

[3분대본]
(내용)
""".strip()


def prompt_full_body_for_batch(
    batch_papers: List[Any],
    batch_index: int,
    total_batches: int,
    kst_date_iso: str,
    kst_date_korean: str
) -> str:
    papers_info = build_papers_info(batch_papers)

    return f"""
오늘은 {kst_date_korean}이며, {kst_date_iso}자 기준으로 서술하십시오.

아래는 상위 선별 논문 배치 {batch_index}/{total_batches}입니다.
(중요: 아래에 포함된 논문만 본문 스크립트를 작성하십시오.)

{papers_info}

중요
- 방송의 도입부와 맺음말을 쓰지 마십시오.
- "첫 번째 논문", "이번 배치", "안녕하세요", "오늘은" 같은 진행 멘트와 순서 멘트를 쓰지 마십시오.
- 오직 각 논문 설명 본문만 출력하십시오.

분량
- 논문 1편당 약 1700~2200자 내외로 상세히 설명하십시오.

언어 규칙
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용하십시오.
- 전문 기술 용어는 번역하지 말고 영어 원어 그대로 사용하십시오.
- 쉼표(,)로 호흡, 마침표(.)로 강조하십시오.
- 공적인 라디오 방송 톤의 존댓말을 유지하십시오.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠)를 사용하지 마십시오.
- "A.", "B.", "첫째", "다음으로", "이어서" 등 구조 또는 순서를 직접적으로 언급하지 마십시오.

출력 형식(반드시 준수)
TITLE: <영문 제목>
BODY:
<본문>

논문과 논문 사이는 빈 줄 2줄로 구분하십시오.
""".strip()


def call_gpt_text(system_text: str, user_text: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens
    )
    return (resp.choices[0].message.content or "").strip()


# =========================
# TTS
# =========================
def synthesize_tts_to_audio(text: str, tts_chunk_chars: int = 2000, overlap: int = 0) -> AudioSegment:
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


def parse_title_body_blocks(text: str) -> List[Tuple[str, str]]:
    t = (text or "").strip()
    if not t:
        return []

    pattern = r"TITLE:\s*(.*?)\s*BODY:\s*(.*?)(?=(?:\n\s*TITLE:)|\Z)"
    matches = re.findall(pattern, t, flags=re.DOTALL | re.IGNORECASE)

    blocks = []
    for title, body in matches:
        tt = title.strip()
        bb = body.strip()
        if tt and bb:
            blocks.append((tt, bb))
    return blocks


def assemble_radio_script(
    full_batches_text: List[str],
    total_papers: int,
    now_kst: datetime.datetime
) -> str:
    date_korean = format_kst_date(now_kst)

    intro = (
        "안녕하세요, 아이알씨브이 랩실의 수석 연구 비서입니다.\n"
        f"지금부터 {date_korean}자 arXiv에 업데이트된 컴퓨터 비전 논문들을 브리핑 드리겠습니다."
    )
    outro = "이상으로 오늘의 전체 브리핑을 마칩니다."

    all_blocks: List[Tuple[str, str]] = []
    for batch_text in full_batches_text:
        all_blocks.extend(parse_title_body_blocks(batch_text))

    script_parts = [intro, ""]
    for i, (title, body) in enumerate(all_blocks, start=1):
        title_tts = sanitize_title_for_tts(title)
        ordinal = number_to_korean_ordinal(i)
        script_parts.append(f"{ordinal} 논문입니다.")
        script_parts.append(f"논문 제목은 {title_tts} 입니다.")
        script_parts.append(body)
        script_parts.append("")

    if len(all_blocks) < total_papers:
        script_parts.append("일부 원고 생성이 누락되어, 생성된 부분까지만 읽었습니다.")
        script_parts.append("")

    script_parts.append(outro)
    return "\n".join(script_parts).strip()


# =========================
# NOTION
# =========================
def notion_rich_link(text: str, url: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": {"content": text, "link": {"url": url}}
    }


def run_bot():
    now_kst = datetime.datetime.now(SEOUL_TZ)
    now_et = now_kst.astimezone(ET_TZ)

    start_et, end_et, announce_date = compute_announce_window_et(now_et)

    kst_date_iso = format_kst_date_iso(now_kst)
    kst_date_korean = format_kst_date(now_kst)

    candidates = fetch_arxiv_candidates_in_window(start_et, end_et)
    if not candidates:
        print("해당 announce window에서 새로 잡힌 논문이 없습니다.")
        return

    prescored = [(p, prescore_text(p)) for p in candidates]
    prescored.sort(key=lambda x: x[1], reverse=True)
    preselected = [p for (p, _) in prescored[:min(PRESELECT_K, len(prescored))]]

    scored: List[ScoredPaper] = []
    for p in preselected:
        scored.append(score_paper(p))

    scored.sort(key=lambda sp: sp.score, reverse=True)
    top = scored[:min(TOPK_SAVE, len(scored))]
    top_papers = [sp.paper for sp in top]

    if not top_papers:
        print("TopK 선별 결과가 비어 있습니다.")
        return

    system_summary = "너는 IRCV 랩실의 수석 연구 비서다. 한국어로 요약과 3분 대본을 공적인 존댓말 톤으로 작성하라."
    system_full = "너는 IRCV 랩실의 수석 연구 비서다. 한국어로 논문 본문 스크립트만 공적인 존댓말 톤으로 작성하라."

    user_summary = prompt_summary_and_3min(top_papers, kst_date_iso, kst_date_korean)
    summary_out = call_gpt_text(system_summary, user_summary, MAX_OUT_TOKENS_SUMMARY)

    if "[3분대본]" in summary_out:
        summary_text = summary_out.split("[3분대본]")[0].replace("[요약]", "").strip()
        audio_script_3min = summary_out.split("[3분대본]")[1].strip()
    else:
        summary_text = summary_out.replace("[요약]", "").strip()
        audio_script_3min = ""

    paper_batches = [top_papers[i:i + BATCH_SIZE_FULL] for i in range(0, len(top_papers), BATCH_SIZE_FULL)]
    total_batches = len(paper_batches)

    full_batches_text: List[str] = []
    for idx, batch in enumerate(paper_batches, start=1):
        user_full = prompt_full_body_for_batch(batch, idx, total_batches, kst_date_iso, kst_date_korean)
        batch_text = call_gpt_text(system_full, user_full, MAX_OUT_TOKENS_FULL_PER_BATCH)
        full_batches_text.append(batch_text)

    audio_script_full = assemble_radio_script(
        full_batches_text,
        total_papers=len(top_papers),
        now_kst=now_kst
    )

    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    today_date = now_kst.strftime("%Y%m%d")
    file_name_full = f"CV_Daily_Briefing_{today_date}.mp3"
    file_name_3min = f"3Min_Summary_{today_date}.mp3"

    full_file_path = os.path.join(audio_dir, file_name_full)
    full_file_path_3min = os.path.join(audio_dir, file_name_3min)

    combined_audio = synthesize_tts_to_audio(
        audio_script_full,
        tts_chunk_chars=TTS_CHUNK_CHARS,
        overlap=TTS_CHUNK_OVERLAP
    )
    combined_audio.export(full_file_path, format="mp3")

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

    page_title = f"{now_kst.strftime('%Y-%m-%d')} 모닝 브리핑 논문 {len(top_papers)}개"

    notion_children = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"기준 날짜: {kst_date_iso} (KST)"}}]}
        },
        {"object": "block", "type": "divider", "divider": {}},
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
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Top 논문 원문 링크"}}]}},
    ]

    for rank, sp in enumerate(top, start=1):
        p = sp.paper
        line = f"{rank}. {p.title}  (score={sp.score:.1f}, author={sp.score_detail['author']:.1f}, content={sp.score_detail['content']:.1f})"
        notion_children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {"type": "text", "text": {"content": line + " "}},
                    {
                        "type": "text",
                        "text": {"content": "PDF", "link": {"url": p.pdf_url}},
                        "annotations": {"bold": True, "color": "default"}
                    }
                ]
            }
        })

    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "요약 & 논문링크": {"title": [{"text": {"content": page_title}}]},
            "날짜": {"date": {"start": now_kst.date().isoformat()}},
            "전체 브리핑": {"rich_text": [notion_rich_link("▶ 전체 브리핑 다운", audio_url)]},
            "3분 요약": {"rich_text": [notion_rich_link("▶ 3분 요약 다운", audio_url_3min)]},
        },
        children=notion_children
    )

    print(f"완료: candidates={len(candidates)}, preselected={len(preselected)}, saved_top={len(top_papers)}")


if __name__ == "__main__":
    run_bot()
