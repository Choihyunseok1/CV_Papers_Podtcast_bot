import os
import arxiv
import openai
from notion_client import Client
import datetime

# 깃허브 금고(Secrets)에서 열쇠 가져오기
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# 도구들 세팅
notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def run_bot():
    # 1. arXiv에서 최신 논문 3개 검색
    search = arxiv.Search(
        query="cat:cs.CV", 
        max_results=3, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    for result in search.results():
        # 2. GPT로 팟캐스트용 3줄 요약 만들기
        prompt = f"논문 제목: {result.title}\n초록: {result.summary}\n이 내용을 연구실 등굣길에 듣기 좋은 친근한 한국어 3줄 요약으로 만들어줘."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        summary_text = response.choices[0].message.content

        # 3. 노션에 등록하기
        notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "이름": {"title": [{"text": {"content": result.title}}]},
                "날짜": {"date": {"start": datetime.date.today().isoformat()}},
                "카테고리": {"select": {"name": "CV"}},
                "요약": {"rich_text": [{"text": {"content": summary_text}}]},
                "오디오": {"url": result.pdf_url} # 우선 PDF 링크로 대체
            }
        )
        print(f"등록 성공: {result.title}")

if __name__ == "__main__":
    run_bot()
