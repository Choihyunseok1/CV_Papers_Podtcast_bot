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

        # 5. GPT-4o에게 '간결한 요약체' 대본 및 요약 요청
        prompt = f"""
        논문 제목: {result.title}
        초록: {result.summary}
        
        위 내용을 바탕으로 다음 두 가지를 작성해 주세요.
        
        
        1. [요약]
        - 노션 기록용 핵심 요약.
        - '-함', '-임' 형태의 짧은 요약체로 3줄 작성. 
        - 불필요한 수식어는 제외하고 핵심 키워드와 데이터 중심으로 구성할 것.
        
        2. [대본]
        - 형식: 라디오 방송의 '오늘의 논문 브리핑' 코너 스크립트.
        - 청취자: 해당 분야에 관심 있는 연구자 및 전문가 그룹.
        - 어조: 정중한 경어체(해요체 혹은 하십시오체). 신뢰감 있고 차분한 톤.
        - 구성 가이드:
            * **도입:** 논문 제목을 언급하며, 이 연구가 학계나 산업계에 던지는 화두를 정중하게 제시할 것.
            * **전개:** 연구의 핵심 가설과 실험 결과를 논리적으로 설명하되, 문어체(~은/는/이/가 등)보다는 말하기 편한 구어체로 다듬을 것. (예: "연구팀은 ~라고 밝혔습니다" 등)
            * **인사이트:** 이 논문이 갖는 학술적 가치나 향후 연구에 미칠 영향력을 전문가적 시각에서 정리해 줄 것.
            * **마무리:** 청취자에게 생각할 거리를 던지며 정중하게 인사를 건넬 것.

        출력시 참고사항
        전문 용어는 가급적 한국어 발음대로 적을 것 (예: CNN -> 씨엔엔)
        끊어 읽기가 필요할 때: 쉼표를 적절히 넣을 것.
        강조하고 싶을 때: 강조하고 싶은 단어 뒤에 한 칸 공백을 더 두거나 문장을 짧게 끊을 것.
        
        출력 형식:
        [요약]
        (요약 내용)
        
        [대본]
        (라디오 스크립트 내용)


        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "너는 IRCV 랩실의 효율적인 연구 비서야."},
                      {"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        
        # [요약]과 [대본] 부분을 분리해서 가져오기
        summary_text = full_text.split("[대본]")[0].replace("[요약]", "").strip()
        audio_script = full_text.split("[대본]")[1].strip()

        # 6. OpenAI TTS로 목소리 생성 (대본을 읽도록 함)
        audio_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="Echo",
            input=audio_script
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
