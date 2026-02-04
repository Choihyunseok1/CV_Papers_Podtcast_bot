import os
import arxiv
import openai
from notion_client import Client
import datetime
from pytz import timezone
from pydub import AudioSegment
import io
import re
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
MAX_OUT_TOKENS_FULL_PER_BATCH = 2800

TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"
TTS_SPEED = 1.25

TTS_CHUNK_CHARS = 2000
TTS_CHUNK_OVERLAP = 0

AUTHOR_SCORE_ENABLED = True

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"

AUTHOR_WEIGHT = 10.0
AUTHOR_HINDEX_CAP = 80.0
AUTHOR_MIN_REQUEST_INTERVAL_SEC = 0.15
AUTHOR_HTTP_TIMEOUT_SEC = 10
AUTHOR_HTTP_RETRIES = 3
AUTHOR_HTTP_BACKOFF_SEC = 0.6

AUTHOR_AGG_METHOD = "max"
AUTHOR_SCORE_SOURCE = "hindex"


_author_cache_by_name = {}
_author_last_request_time = 0.0


def _semantic_scholar_headers():
    headers = {
        "Accept": "application/json"
    }
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers


def _rate_limit_sleep():
    global _author_last_request_time
    now = time.time()
    elapsed = now - _author_last_request_time
    if elapsed < AUTHOR_MIN_REQUEST_INTERVAL_SEC:
        time.sleep(AUTHOR_MIN_REQUEST_INTERVAL_SEC - elapsed)
    _author_last_request_time = time.time()


def _http_get_json(url, params=None, timeout_sec=10, retries=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            _rate_limit_sleep()
            r = requests.get(
                url,
                params=params,
                headers=_semantic_scholar_headers(),
                timeout=timeout_sec
            )
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e

        time.sleep(AUTHOR_HTTP_BACKOFF_SEC * attempt)

    raise last_err


def _ss_find_author_id_by_name(author_name):
    author_name = (author_name or "").strip()
    if not author_name:
        return None

    cache = _author_cache_by_name.get(author_name)
    if cache and "authorId" in cache:
        return cache["authorId"]

    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/author/search"
    params = {
        "query": author_name,
        "limit": 1,
        "fields": "authorId,name"
    }
    data = _http_get_json(
        url,
        params=params,
        timeout_sec=AUTHOR_HTTP_TIMEOUT_SEC,
        retries=AUTHOR_HTTP_RETRIES
    )
    items = data.get("data", []) if isinstance(data, dict) else []
    if not items:
        _author_cache_by_name[author_name] = {"authorId": None}
        return None

    author_id = items[0].get("authorId")
    _author_cache_by_name[author_name] = {"authorId": author_id}
    return author_id


def _ss_get_author_metrics(author_id):
    if not author_id:
        return None

    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/author/{author_id}"
    params = {
        "fields": "name,hIndex,citationCount,paperCount"
    }
    data = _http_get_json(
        url,
        params=params,
        timeout_sec=AUTHOR_HTTP_TIMEOUT_SEC,
        retries=AUTHOR_HTTP_RETRIES
    )
    if not isinstance(data, dict):
        return None

    return {
        "name": data.get("name"),
        "hIndex": float(data.get("hIndex") or 0.0),
        "citationCount": float(data.get("citationCount") or 0.0),
        "paperCount": float(data.get("paperCount") or 0.0),
    }


def get_author_score_for_paper(arxiv_paper):
    if not AUTHOR_SCORE_ENABLED:
        return 0.0, []

    authors = getattr(arxiv_paper, "authors", None) or []
    authors = authors[:2]   # 1저자 + 2저자만 사용
    if not authors:
        return 0.0, []

    author_scores = []
    author_debug = []

    for a in authors:
        name = str(a).strip()
        if not name:
            continue

        if name in _author_cache_by_name and "metrics" in _author_cache_by_name[name]:
            metrics = _author_cache_by_name[name]["metrics"]
        else:
            try:
                author_id = _ss_find_author_id_by_name(name)
                metrics = _ss_get_author_metrics(author_id) if author_id else None
            except Exception:
                metrics = None

            if name not in _author_cache_by_name:
                _author_cache_by_name[name] = {}
            _author_cache_by_name[name]["metrics"] = metrics

        if not metrics:
            author_debug.append({"name": name, "hIndex": 0.0, "citationCount": 0.0, "paperCount": 0.0})
            author_scores.append(0.0)
            continue

        hidx = float(metrics.get("hIndex") or 0.0)
        ctc = float(metrics.get("citationCount") or 0.0)
        pc = float(metrics.get("paperCount") or 0.0)

        if AUTHOR_SCORE_SOURCE == "hindex":
            raw = hidx
        elif AUTHOR_SCORE_SOURCE == "citations":
            raw = ctc
        elif AUTHOR_SCORE_SOURCE == "papers":
            raw = pc
        else:
            raw = hidx

        author_debug.append({"name": name, "hIndex": hidx, "citationCount": ctc, "paperCount": pc})
        author_scores.append(raw)

    if not author_scores:
        return 0.0, author_debug

    if AUTHOR_AGG_METHOD == "mean":
        agg = sum(author_scores) / max(1, len(author_scores))
    else:
        agg = max(author_scores)

    norm = min(max(agg, 0.0), AUTHOR_HINDEX_CAP) / max(1.0, AUTHOR_HINDEX_CAP)

    return float(norm), author_debug


def medical_penalty_score(title, abstract):
    text = f"{title or ''} {abstract or ''}".lower()

    keywords = [
        "medical", "clinical", "patient", "disease",
        "cancer", "tumor", "lesion",
        "pathology", "histopathology", "radiology",
        "ct", "mri", "pet", "ultrasound",
        "organ", "tissue", "cell",
        "diagnosis", "screening", "prognosis",
        "pancreatic", "lung cancer", "breast cancer"
    ]

    hit = 0
    for k in keywords:
        if k in text:
            hit += 1

    return 25 if hit >= 2 else 0


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
- 각 논문별로 제목을 언급하고, '-함', '-임' 형태의 짧은 요약체로 3줄씩 작성할 것
- 한 줄이 끝나면 반드시 엔터로 구분해서 보기 편하게 만들 것
- 각 논문 요약 시작시 '1. (논문제목)' 식으로 앞에 번호만 붙여 진행할 것
- 논문들 사이는 줄바꿈으로 구분할 것.

2. [3분대본]
- "시간이 없으신 분들을 위한 3분 핵심 요약입니다"로 시작할 것.
- 모든 논문을 빠짐없이 포함할 것.
- 각 논문 제목을 말한 뒤 , 논문 당 약 400자 내외로 설명할 것.
- 전체 길이는 약 3분(±15초) 분량이 되도록 조절할 것.
- 논문 수가 많을 경우, 각 논문의 설명 길이를 자동으로 줄여서 전체 분량을 유지할 것.
- 분량이 부족하더라도 일부 논문을 생략하지 말 것.
- 모든 논문을 최소한 한 단락 이상 설명할 것.
- 논문의 공식 제목은 반드시 영문으로 표기하되, 제목의 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿀 것.
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용할 것.
- 전문 기술 용어(diffusion, transformer, attention, self-attention, cross-attention, latent, encoder, decoder, backbone, head, neck, pipeline, architecture, framework, module, block, layer, stage, feature, representation, embedding, token, patch, pixel, resolution, scale, multi-scale, spatial, temporal, semantic, instance, object, bounding box, mask, classification, regression, detection, segmentation, tracking, matching, retrieval, generation, reconstruction, prediction, training, inference, optimization, loss, objective, gradient, backpropagation, scheduler, warmup, regularization, overfitting, underfitting, convergence, likelihood, log-likelihood, prior, posterior, sampling, denoising, noise, variance, distribution, gaussian, entropy, kl-divergence, dataset, benchmark, metric, accuracy, precision, recall, f-score, mean average precision, intersection over union, foundation model, large-scale, multi-modal, vision-language, prompt, prompting, alignment, zero-shot, few-shot, in-context learning, parameter-efficient tuning, point cloud, voxel, mesh, depth, pose, camera, ray, rendering, video, frame, motion, optical flow, reinforcement learning, policy, value function, reward, exploration, exploitation, environment, agent, state, action, episode, timestep, imitation learning, self-supervised learning, supervised learning, unsupervised learning, contrastive learning, pretraining, fine-tuning, transfer learning, curriculum learning, data augmentation, normalization, batch normalization, layer normalization, residual connection, skip connection, attention map, positional encoding, query, key, value, softmax, temperature, logits, probability, score, confidence, threshold, calibration, robustness, generalization, scalability, efficiency, latency, throughput, memory, parameter, hyperparameter, initialization, seed, reproducibility, ablation study, baseline, state-of-the-art, sota, comparison, improvement, gain, trade-off, limitation, future work 등)는 번역하지 말고 반드시 영어 원어 그대로 사용할 것.
- 쉼표(,)를 충분히 사용해 호흡 지점을 표시할 것.
- 동료 연구자에게 설명하듯 차분한 구어체.
- 전체 브리핑은 반드시 공적인 라디오 방송 톤의 존댓말로 작성할 것.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠)는 사용하지 말 것.
- 연구 비서가 공식적으로 설명하는 말투를 유지할 것.

[마무리 규칙]
- 모든 논문 설명이 끝난 뒤, 반드시 아래 톤의 아웃트로 멘트를 한 문단으로 추가할 것.
- 감사 인사나 일상적인 인삿말은 사용하지 말 것.
- 더 자세한 내용이 전체 브리핑에 있다는 점을 자연스럽게 안내할 것.

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
- 논문 1편당 약 2200자 내외로 상세히 설명하세요.

구조 (내부 참고용):
A. 문제의식과 배경
B. 핵심 아이디어 한 줄 요약과 의미
C. 방법을 단계적으로 설명
D. 실험과 결과의 경향
E. 한계와 추후 과제
F. 실전 감상 포인트 2개

언어 규칙:
- 쉼표(,)로 호흡, 마침표(.)로 강조.
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용할 것.
- 동료 연구자에게 설명하듯 차분한 구어체.
- 오디오 스크립트에서는 절대 "A.", "B.", "첫째", "다음으로", "이어서" 등 구조나 순서를 직접적으로 언급하지 마세요.
- 전체 브리핑은 반드시 공적인 라디오 방송 톤의 존댓말로 작성할 것.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠)는 사용하지 말 것.
- 연구 비서가 공식적으로 설명하는 말투를 유지할 것.
- 전문 기술 용어(diffusion, transformer, attention, self-attention, cross-attention, latent, encoder, decoder, backbone, head, neck, pipeline, architecture, framework, module, block, layer, stage, feature, representation, embedding, token, patch, pixel, resolution, scale, multi-scale, spatial, temporal, semantic, instance, object, bounding box, mask, classification, regression, detection, segmentation, tracking, matching, retrieval, generation, reconstruction, prediction, training, inference, optimization, loss, objective, gradient, backpropagation, scheduler, warmup, regularization, overfitting, underfitting, convergence, likelihood, log-likelihood, prior, posterior, sampling, denoising, noise, variance, distribution, gaussian, entropy, kl-divergence, dataset, benchmark, metric, accuracy, precision, recall, f-score, mean average precision, intersection over union, foundation model, large-scale, multi-modal, vision-language, prompt, prompting, alignment, zero-shot, few-shot, in-context learning, parameter-efficient tuning, point cloud, voxel, mesh, depth, pose, camera, ray, rendering, video, frame, motion, optical flow, reinforcement learning, policy, value function, reward, exploration, exploitation, environment, agent, state, action, episode, timestep, imitation learning, self-supervised learning, supervised learning, unsupervised learning, contrastive learning, pretraining, fine-tuning, transfer learning, curriculum learning, data augmentation, normalization, batch normalization, layer normalization, residual connection, skip connection, attention map, positional encoding, query, key, value, softmax, temperature, logits, probability, score, confidence, threshold, calibration, robustness, generalization, scalability, efficiency, latency, throughput, memory, parameter, hyperparameter, initialization, seed, reproducibility, ablation study, baseline, state-of-the-art, sota, comparison, improvement, gain, trade-off, limitation, future work 등)는 번역하지 말고 반드시 영어 원어 그대로 사용할 것.

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

        if i == 1:
            transition = "오늘 첫 번째로 살펴볼 논문입니다."
        else:
            transition = f"계속해서 {i}번째 논문을 보겠습니다."

        script_parts.append(transition)
        script_parts.append(f"논문 제목은 {title_tts} 입니다.")
        script_parts.append(body)
        script_parts.append("")

    if len(all_blocks) < total_papers:
        script_parts.append("일부 논문은 요약 중심으로 간략히 다뤄졌습니다.")
        script_parts.append("")

    script_parts.append(outro)
    return "\n".join(script_parts).strip()


def get_submission_window_et(now_et):
    wd = now_et.weekday()
    
    if wd in (4, 5):
        return None, None

    today_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)

    def at_14(d):
        return d.replace(hour=14, minute=0, second=0, microsecond=0)

    if wd == 1:
        end_et = at_14(today_et)
        start_et = end_et - datetime.timedelta(days=1)
        return start_et, end_et

    if wd == 2:
        end_et = at_14(today_et)
        start_et = end_et - datetime.timedelta(days=1)
        return start_et, end_et

    if wd == 3:
        end_et = at_14(today_et)
        start_et = end_et - datetime.timedelta(days=1)
        return start_et, end_et

    if wd == 6:
        end_et = at_14(today_et - datetime.timedelta(days=2))
        start_et = end_et - datetime.timedelta(days=1)
        return start_et, end_et

    if wd == 0:
        end_et = at_14(today_et)
        start_et = end_et - datetime.timedelta(days=3)
        return start_et, end_et

    return None, None


def run_bot():
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    et = ZoneInfo("America/New_York")
    now_et = datetime.datetime.now(et)

    #----------------------------------------------------------------- 디버깅용 날짜 임의 조작 ----------------------------------------------------------
    now_et = datetime.datetime(2026, 2, 3, 20, 0, 0, tzinfo=et)
    #--------------------------------------------------------------------------------------------------------------------------------------------------

    window_start_et, window_end_et = get_submission_window_et(now_et)
    print("Submission window (ET)")
    print(f"window_start_et = {window_start_et}")
    print(f"window_end_et   = {window_end_et}")

    if window_start_et is None:
        print("오늘은 arXiv announce가 없는 날(ET 기준 금 토)이라 종료합니다.")
        return

    seoul_tz = timezone("Asia/Seoul")
    now = datetime.datetime.now(seoul_tz)

    search = arxiv.Search(
        query="cat:cs.CV",
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    print("arXiv search config")
    print("query=cat:cs.CV")
    print("max_results=200")
    print("sort_by=SubmittedDate(desc)")

    arxiv_client = arxiv.Client()

    candidates = []
    total_scanned = 0
    for p in arxiv_client.results(search):
        total_scanned += 1
        p_submitted_et = p.published.astimezone(et)

        print(f"[SCAN] {p.title[:120]}")
        print(f"       submitted_et = {p_submitted_et}")

        if window_start_et <= p_submitted_et < window_end_et:
            print("    -> IN WINDOW")
            candidates.append(p)
        else:
            print("    -> OUT OF WINDOW")

    print(f"Total scanned papers = {total_scanned}")
    print(f"Papers in submission window = {len(candidates)}")

    if not candidates:
        print("해당 submission window에서 새로 올라온 논문이 없습니다.")
        print(f"window(ET): {window_start_et} ~ {window_end_et}")
        print("No paper satisfied:")
        print("window_start_et <= published_et < window_end_et")
        return

    scored = []
    for p in candidates:
        # 1. 의료/바이오 논문은 즉시 제외 (저자 조회 안 함)
        if medical_penalty_score(p.title, p.summary) > 0:
            continue
    
        # 2. 저자 점수만 사용 (0~1 정규화 값)
        author_norm, author_debug = (0.0, [])
        if AUTHOR_SCORE_ENABLED:
            author_norm, author_debug = get_author_score_for_paper(p)
    
        final_score = author_norm
    
        # 3. 디버그 로그 (의미 있는 정보만)
        print("[SCORE]")
        print(f"[title = {p.title[:80]}")
        print(f"author_score = {final_score:.4f}")
    
        if AUTHOR_SCORE_ENABLED and author_debug:
            for a in author_debug[:2]:  # 1,2저자만 출력
                print(
                    f"    {a.get('name')} "
                    f"    hIndex={a.get('hIndex')} "
                    f"    citations={a.get('citationCount')} "
                    f"    papers={a.get('paperCount')}"
                )
    
        scored.append((final_score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    valid_papers = [x[3] for x in scored[:10]]

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
    page_title = f"{now.strftime('%Y-%m-%d')} 브리핑"

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
            "요약 & 논문링크": {"title": [{"text": {"content": page_title}}]},
            "날짜": {"date": {"start": now.date().isoformat()}},
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

    print(f"통합 브리핑 생성 완료: {len(valid_papers)}개의 논문")


if __name__ == "__main__":
    run_bot()
