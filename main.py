import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# โหลดค่า Environment จากไฟล์ .env
load_dotenv()

# 1. กำหนดโครงสร้าง State สำหรับรับส่งข้อมูลระหว่าง Agent
class AgentState(TypedDict):
    query: str
    search_keywords: str
    context: str
    final_answer: str

# 2. สร้าง Custom Tool สำหรับ RAG ค้นหาข้อมูลจาก knowledge_base.txt
@tool
def search_knowledge_base(keywords: str) -> str:
    """ค้นหาคำสำคัญในไฟล์ knowledge_base.txt และส่งคืนย่อหน้าที่เกี่ยวข้อง"""
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        search_terms = keywords.lower().split()
        results = []
        
        # ทำ Basic Keyword Search
        for p in paragraphs:
            if any(term in p.lower() for term in search_terms):
                results.append(p)
                
        if not results:
            return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้"
        
        return "\n\n---\n\n".join(results)
    except FileNotFoundError:
        return "Error: ไม่พบไฟล์ knowledge_base.txt"

# 3. ตั้งค่า Google Gemini LLM
# สามารถเลือกใช้โมเดล 'gemini-1.5-flash' (เร็วและประหยัดโควตา) หรือ 'gemini-1.5-pro'
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",  # เปลี่ยนมาใช้โมเดลที่มีอยู่ในระบบ
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY") 
)

# 4. Agent Node 1: Data Retriever (สกัดคีย์เวิร์ดและดึงข้อมูล)
def data_retriever_node(state: AgentState):
    query = state["query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Data Retriever. Extract 1-3 core keywords from the user's query to search a text database. Return ONLY the keywords separated by spaces. Do not translate."),
        ("user", "{query}")
    ])
    
    keyword_chain = prompt | llm
    keywords = keyword_chain.invoke({"query": query}).content.strip()
    
    # เรียกใช้ Tool ดึงข้อมูลดิบ
    retrieved_context = search_knowledge_base.invoke(keywords)
    
    return {"search_keywords": keywords, "context": retrieved_context}

# 5. Agent Node 2: Report Generator (สังเคราะห์คำตอบ)
def report_generator_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Report Generator. Use ONLY the provided Text Snippets to answer the User Query. If the snippets do not contain the answer, simply state that the information is unavailable. Format the output nicely using bullet points if applicable. Write in the same language as the user query."),
        ("user", "User Query: {query}\n\nText Snippets:\n{context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "context": state["context"]
    })
    
    return {"final_answer": response.content}

# 6. ประกอบ Graph ด้วย LangGraph (Sequential Orchestration)
workflow = StateGraph(AgentState)

workflow.add_node("data_retriever", data_retriever_node)
workflow.add_node("report_generator", report_generator_node)

workflow.set_entry_point("data_retriever")
workflow.add_edge("data_retriever", "report_generator")
workflow.add_edge("report_generator", END)

app = workflow.compile()

# 7. ทดสอบการทำงาน
if __name__ == "__main__":
    import time  # ตรวจสอบให้แน่ใจว่า import time ไว้แล้ว

    print("ระบบ Agentic AI พร้อมใช้งาน (พิมพ์ 'exit' เพื่อออก)\n")
    
    test_queries = [
        "เบิกค่าที่พักสำหรับเดินทางไปต่างประเทศได้สูงสุดคืนละเท่าไหร่?",
        "ถ้าต้องการลาพักผ่อนติดกัน 4 วัน ต้องแจ้งล่วงหน้ากี่วัน?",
        "บริษัทมีสวัสดิการสนับสนุนค่าฟิตเนสให้พนักงานไหม เดือนละเท่าไหร่?",
        "หากทำโน้ตบุ๊กของบริษัทพังจากความประมาท พนักงานต้องรับผิดชอบค่าซ่อมอย่างไร?"
    ]
    
    for q in test_queries:
        print(f"User Query: {q}")
        result = app.invoke({"query": q})
        print(f"Keywords Searched: [{result['search_keywords']}]")
        print(f"Report Generator Output:\n{result['final_answer']}")
        print("-" * 60 + "\n")
        
        # เพิ่มการหน่วงเวลา 15 วินาทีเพื่อป้องกัน API Rate Limit
        print("(รอ 15 วินาทีเพื่อป้องกันการเรียก API เกินโควตา...)")
        time.sleep(15)