import os
import arxiv
import openai
from notion_client import Client
import datetime
from pytz import timezone
from pydub import AudioSegment
import io
import re
import json
from zoneinfo import ZoneInfo
import time
import requests


NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GITHUB_USER = "Choihyunseok1"
GITHUB_REPO = "CV_Papers_Podtcast_bot"

notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

BATCH_SIZE_FULL = 2
MAX_OUT_TOKENS_SUMMARY = 4000
MAX_OUT_TOKENS_FULL_PER_PAPER = 4000

TOP_PAPERS_LIMIT = 10
MAX_RESULTS = 200

AUTHOR_SCORE_ENABLED = True
AUTHOR_SCORE_LIMIT = 2
AUTHOR_SCORE_HINDEX_CAP = 80

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

WINDOW_DAYS = 1

ET = timezone("US/Eastern")

NON_CV_KEYWORDS = [
    "medical",
    "medicine",
    "clinical",
    "patient",
    "radiology",
    "pathology",
    "ct",
    "mri",
    "ultrasound",
    "ecg",
    "eeg",
    "diagnosis",
    "diagnostic",
    "disease",
    "tumor",
    "cancer",
    "biomedical",
    "bioinformatics",
    "genomics",
    "protein",
    "cell",
    "molecular",
    "drug",
    "pharma",
    "surgery",
]

CV_KEYWORDS = [
    "computer vision",
    "vision",
    "image",
    "video",
    "segmentation",
    "detection",
    "recognition",
    "tracking",
    "3d",
    "depth",
    "stereo",
    "pose",
    "slam",
    "reconstruction",
    "gaussian splatting",
    "diffusion",
    "generative",
    "multimodal",
    "vqa",
    "caption",
    "vlm",
    "transformer",
    "self-supervised",
    "representation learning",
]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def is_today_published_in_et(published_dt_utc, window_start_et, window_end_et):
    if published_dt_utc is None:
        return False
    published_et = published_dt_utc.astimezone(ET)
    return (window_start_et <= published_et < window_end_et)


def contains_any_keyword(text, keywords):
    t = (text or "").lower()
    for kw in keywords:
        if kw.lower() in t:
            return True
    return False


def medical_penalty_score(title, abstract):
    text = f"{title}\n{abstract}".lower()
    for kw in NON_CV_KEYWORDS:
        if kw in text:
            return 1
    return 0


def cv_relevance_penalty_score(title, abstract):
    text = f"{title}\n{abstract}".lower()
    cv_hit = 0
    for kw in CV_KEYWORDS:
        if kw in text:
            cv_hit += 1

    if cv_hit < 2:
        return 1
    return 0


def get_semantic_scholar_author(name: str):
    q = normalize_text(name)
    if not q:
        return None

    url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {"query": q, "limit": 1, "fields": "name,hIndex,citationCount,paperCount"}

    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        if not data.get("data"):
            return None
        return data["data"][0]
    except Exception:
        return None


def normalize_hindex(hindex: int, cap: int = AUTHOR_SCORE_HINDEX_CAP):
    if hindex is None:
        return 0.0
    try:
        h = float(hindex)
    except Exception:
        return 0.0
    if h < 0:
        h = 0.0
    if h > cap:
        h = cap
    return h / float(cap)


def get_author_score_for_paper(paper):
    authors = paper.authors or []
    debug = []
    scores = []

    for a in authors[:AUTHOR_SCORE_LIMIT]:
        name = getattr(a, "name", None) or str(a)
        info = get_semantic_scholar_author(name)
        if info:
            h = info.get("hIndex")
            scores.append(normalize_hindex(h))
            debug.append(info)
        else:
            scores.append(0.0)
            debug.append({"name": name, "hIndex": None, "citationCount": None, "paperCount": None})

    if not scores:
        return 0.0, debug

    return max(scores), debug


def build_papers_info(valid_papers):
    papers_info = ""
    for i, p in enumerate(valid_papers):
        papers_info += f"\n[{i+1}] {p.title}\n"
        papers_info += f"Abstract: {p.summary}\n"
        papers_info += f"PDF: {p.pdf_url}\n"
    return papers_info


def prompt_summary_and_3min(valid_papers):
    papers_info = build_papers_info(valid_papers)

    user_text = f"""
아래는 오늘 arXiv cs.CV에 업데이트된 주요 논문 10개 정보다.
요구사항:
- 1) 전체 흐름/오늘의 기술 트렌드를 7~10줄로 요약해라(한국어).
- 2) 이어서 라디오 진행자가 말할 수 있는 3분 분량 스크립트를 작성해라(한국어).
- 3) 각 논문은 1~2문장씩만 짚고, 연결 흐름이 자연스럽게.
- 4) 너무 논문 나열하지 말고, 큰 흐름을 강조해라.

논문 목록:
{papers_info}
""".strip()
    return user_text


def prompt_full_script_for_each_paper(paper):
    user_text = f"""
아래 논문 하나에 대해서, 라디오 진행자가 읽을 수 있는 "논문 본문 스크립트"만 작성해줘.
요구사항:
- 한국어
- 너무 길지 않게(1~2분 분량)
- 핵심 아이디어, 방법, 장점, 한계, 앞으로의 방향을 포함

Title: {paper.title}
Abstract: {paper.summary}
PDF: {paper.pdf_url}
""".strip()
    return user_text


def chunk_text_by_chars(text, chunk_chars=2000, overlap=0):
    text = text or ""
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap if overlap > 0 else end
    return chunks


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



def call_gpt_json(system_text, user_text, max_tokens=2000, retries=2, sleep_sec=1.0):
    """
    LLM에게 JSON만 출력하도록 요청하고 파싱해서 dict로 반환.
    - 다른 코드 로직에 영향 주지 않기 위해, 실패 시 빈 dict 반환(로그 출력).
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp_text = call_gpt_text(system_text, user_text, max_tokens=max_tokens)
            cleaned = resp_text.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            if "{" in cleaned and "}" in cleaned:
                cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
            return json.loads(cleaned)
        except Exception as e:
            last_err = e
            print(f"[LLM_JSON] parse_failed attempt={attempt} err={e}")
            if attempt < retries:
                time.sleep(sleep_sec)
    print(f"[LLM_JSON] giving_up err={last_err}")
    return {}


def get_arxiv_id_from_pdf_url(pdf_url: str) -> str:
    """
    예: https://arxiv.org/pdf/2602.03414v1 -> 2602.03414v1
        https://arxiv.org/pdf/2602.03414v1.pdf -> 2602.03414v1
    """
    if not pdf_url:
        return ""
    m = re.search(r"/pdf/([0-9]+\.[0-9]+v[0-9]+)", pdf_url)
    if m:
        return m.group(1)
    m = re.search(r"/pdf/([0-9]+\.[0-9]+)", pdf_url)
    if m:
        return m.group(1)
    return ""


def build_today_trend_themes_via_llm(papers, max_titles=160, n_themes=10):
    titles = []
    for p in papers[:max_titles]:
        t = (p.title or "").strip()
        if t:
            titles.append(t)

    titles_block = "\n".join([f"- {t}" for t in titles])

    system_text = "너는 컴퓨터 비전(cs.CV) 연구 트렌드를 요약하는 분석가다. 반드시 JSON만 출력한다."
    user_text = f"""
아래는 오늘 arXiv cs.CV에 올라온 논문 제목 목록이다(의료 논문은 이미 제외됨).
이 목록만 보고, '오늘의 CV 트렌드 테마'를 {n_themes}개 뽑아라.

출력은 반드시 아래 JSON 스키마를 만족해야 한다. 다른 텍스트 금지.

{{
  "themes": [
    {{
      "name": "짧은 테마명",
      "keywords": ["키워드/구문1","키워드/구문2","키워드/구문3"],
      "one_line": "왜 오늘 트렌드인지 1줄 설명"
    }}
  ]
}}

규칙:
- themes 길이는 정확히 {n_themes}
- keywords는 각 테마당 3~6개, 너무 일반적인 단어(예: model, method, learning)는 피함
- 키워드는 제목에 실제로 등장하거나 강하게 연상되는 구문 위주로

제목 목록:
{titles_block}
""".strip()

    data = call_gpt_json(system_text, user_text, max_tokens=2000)
    themes = data.get("themes", []) if isinstance(data, dict) else []
    if not isinstance(themes, list):
        themes = []

    clean_themes = []
    for th in themes:
        if not isinstance(th, dict):
            continue
        name = (th.get("name") or "").strip()
        kws = th.get("keywords") or []
        one_line = (th.get("one_line") or "").strip()
        if name and isinstance(kws, list) and len(kws) >= 1:
            clean_themes.append({
                "name": name,
                "keywords": [str(x).strip() for x in kws if str(x).strip()],
                "one_line": one_line
            })
    return clean_themes


def compute_trend_match_score(title: str, abstract: str, themes) -> float:
    text = f"{title}\n{abstract}".lower()
    score = 0.0
    for th in themes:
        kws = th.get("keywords", [])
        for kw in kws:
            k = str(kw).strip().lower()
            if not k:
                continue
            if k in text:
                score += 1.0
    return score


def select_topk_by_trend(papers, themes, topk=50, min_keep=30):
    scored_local = []
    for p in papers:
        s = compute_trend_match_score(p.title or "", p.summary or "", themes)
        scored_local.append((s, p))

    scored_local.sort(key=lambda x: x[0], reverse=True)

    selected = [p for (s, p) in scored_local[:topk]]

    if len(selected) < min_keep:
        selected = [p for (_, p) in scored_local[:min_keep]]

    return selected, scored_local[:topk]


def pick_top10_via_llm(candidates_topk, n_pick=10):
    items = []
    for p in candidates_topk:
        arxiv_id = get_arxiv_id_from_pdf_url(getattr(p, "pdf_url", "") or "")
        title = (p.title or "").strip()
        abstract = (p.summary or "").strip()
        abstract = abstract[:900]
        items.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract
        })

    system_text = "너는 IRCV 랩실의 수석 연구 비서다. 반드시 JSON만 출력한다."
    user_text = f"""
아래 후보 논문 목록에서 '오늘 업데이트된 cs.CV 논문 중 주요 논문 Top{n_pick}'을 선정해라.
출력은 제목만 뽑되, 내부 디버깅을 위해 1줄 이유도 함께 넣어라.

선정 기준은 네가 판단하되, 대략 다음을 우선한다:
- 새로운 문제 설정/새 태스크
- 커뮤니티 파급력(다음 연구가 참조하기 쉬운 자원/프레임워크/벤치)
- 비디오/멀티모달/3D/효율/생성 등 현재 활발한 흐름과의 연결
- 단순 응용/특정 도메인에만 갇힌 논문은 상대적으로 후순위

반드시 아래 JSON 형식만 출력:
{{
  "top10": [
    {{
      "arxiv_id": "2602.03414v1",
      "title": "논문 제목",
      "reason": "1줄 이유"
    }}
  ],
  "criteria": ["기준1","기준2","기준3","기준4"]
}}

규칙:
- top10 길이는 정확히 {n_pick}
- arxiv_id는 후보에 있는 것만 사용
- title은 후보 title을 그대로 사용
- 다른 텍스트 금지

후보 목록(JSON):
{json.dumps(items, ensure_ascii=False)}
""".strip()

    data = call_gpt_json(system_text, user_text, max_tokens=2200)
    top10 = data.get("top10", []) if isinstance(data, dict) else []
    if not isinstance(top10, list):
        top10 = []
    return data, top10


def synthesize_tts_to_audio(text, tts_chunk_chars=2000, overlap=0):
    chunks = chunk_text_by_chars(text, chunk_chars=tts_chunk_chars, overlap=overlap)
    combined = AudioSegment.empty()
    for chunk in chunks:
        audio_part = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=chunk
        )
        combined += AudioSegment.from_file(io.BytesIO(audio_part.content), format="mp3")
    return combined


def upload_to_notion(date_korean, summary_text, full_scripts, valid_papers):
    children = []

    children.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "모닝 브리핑"}}]}
    })
    children.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": summary_text}}]}
    })

    children.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 본문 스크립트"}}]}
    })

    for i, script in enumerate(full_scripts):
        children.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": f"Paper {i+1}"}}]}
        })
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": script}}]}
        })

    children.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 원문 링크"}}]}
    })

    for i, p in enumerate(valid_papers):
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
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
            "Name": {"title": [{"text": {"content": f"{date_korean}자 모닝 브리핑"}}]},
        },
        children=children
    )


def main():
    now_et = datetime.datetime.now(tz=ET)
    window_end_et = now_et.replace(hour=14, minute=0, second=0, microsecond=0)
    window_start_et = window_end_et - datetime.timedelta(days=WINDOW_DAYS)

    print(f"[DEBUG] window_start_et = {window_start_et}")
    print(f"[DEBUG] window_end_et   = {window_end_et}")
    print("window_start_et <= published_et < window_end_et")

    date_korean = window_end_et.strftime("%Y-%m-%d")

    search = arxiv.Search(
        query="cat:cs.CV",
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results = list(search.results())

    print(f"[DEBUG] fetched_results = {len(results)} (max_results={MAX_RESULTS})")

    candidates = []
    for r in results:
        if is_today_published_in_et(r.published, window_start_et, window_end_et):
            candidates.append(r)

    print(f"[DEBUG] candidates_in_window = {len(candidates)}")

    if not candidates:
        print("오늘 제출된 cs.CV 논문이 없습니다(ET 기준 window).")
        return

    scored = []
    filtered_candidates = []
    for p in candidates:
        # 1) 의료/바이오 논문은 즉시 제외 (저자 조회 안 함)
        med_pen = medical_penalty_score(p.title, p.summary)
        if med_pen > 0:
            print("[FILTER] medical -> skip")
            print(f"         title = {p.title[:80]}")
            continue

        # 2) CV 관련성 낮으면 즉시 제외 (저자 조회 안 함)
        cv_pen = cv_relevance_penalty_score(p.title, p.summary)
        if cv_pen > 0:
            print("[FILTER] low_cv_relevance -> skip")
            print(f"         title = {p.title[:80]}")
            continue

        # 3) (변경) LLM 기반 트렌드/Top10 선정용으로 후보만 모아둠
        filtered_candidates.append(p)
        print("[PASS] candidate_kept")
        print(f"       title = {p.title[:80]}")

    # (변경) 1) 오늘 트렌드 테마/키워드 추출 -> 2) 로컬 TopK 축소 -> 3) LLM 최종 Top10
    if not filtered_candidates:
        print("필터(의료/비CV) 이후 남은 논문이 없습니다.")
        return

    print(f"[CANDIDATES] after_filters = {len(filtered_candidates)}")

    themes = build_today_trend_themes_via_llm(filtered_candidates, max_titles=160, n_themes=10)
    print("[TREND_THEMES]")
    for i, th in enumerate(themes[:10]):
        print(f"  {i+1}. {th.get('name')}")
        kws = th.get("keywords", [])[:6]
        if kws:
            print(f"     keywords: {', '.join(kws)}")
        one_line = th.get("one_line", "")
        if one_line:
            print(f"     note: {one_line}")

    topk_candidates, topk_debug = select_topk_by_trend(filtered_candidates, themes, topk=50, min_keep=30)
    print(f"[TOPK] selected_for_llm = {len(topk_candidates)}")
    for s, p in topk_debug[:10]:
        print(f"  score={s:.1f} title={p.title[:80]}")

    top10_data, top10_list = pick_top10_via_llm(topk_candidates, n_pick=10)

    id_to_paper = {}
    for p in topk_candidates:
        pid = get_arxiv_id_from_pdf_url(getattr(p, "pdf_url", "") or "")
        if pid:
            id_to_paper[pid] = p

    chosen = []
    seen = set()
    for item in top10_list:
        pid = str(item.get("arxiv_id", "")).strip()
        if pid in id_to_paper and pid not in seen:
            chosen.append(id_to_paper[pid])
            seen.add(pid)

    if len(chosen) < 10:
        for p in topk_candidates:
            pid = get_arxiv_id_from_pdf_url(getattr(p, "pdf_url", "") or "")
            key = pid or p.title
            if key in seen:
                continue
            chosen.append(p)
            seen.add(key)
            if len(chosen) >= 10:
                break

    valid_papers = chosen[:10]

    if not valid_papers:
        print("필터(의료/비CV) 이후 남은 논문이 없습니다.")
        return

    system_summary = "너는 IRCV 랩실의 수석 연구 비서이자 AI 전문 라디오 진행자야. 한국어로 요약과 3분 대본을 작성해줘."
    system_full = "너는 IRCV 랩실의 수석 연구 비서이자 AI 전문 라디오 진행자야. 한국어로 논문 본문 스크립트만 작성해줘."

    user_summary = prompt_summary_and_3min(valid_papers)
    summary_text = call_gpt_text(system_summary, user_summary, MAX_OUT_TOKENS_SUMMARY)
    print("[GPT] summary+3min completed")

    full_scripts = []
    for i in range(0, len(valid_papers), BATCH_SIZE_FULL):
        batch = valid_papers[i:i + BATCH_SIZE_FULL]
        for p in batch:
            user_full = prompt_full_script_for_each_paper(p)
            script = call_gpt_text(system_full, user_full, MAX_OUT_TOKENS_FULL_PER_PAPER)
            full_scripts.append(script)
            print(f"[GPT] full_script completed: {p.title[:50]}")
        time.sleep(0.5)

    upload_to_notion(date_korean, summary_text, full_scripts, valid_papers)
    print("[NOTION] upload completed")

    audio = synthesize_tts_to_audio(summary_text, tts_chunk_chars=2000, overlap=0)
    audio.export("morning_briefing.mp3", format="mp3")
    print("[TTS] morning_briefing.mp3 created")


if __name__ == "__main__":
    main()
