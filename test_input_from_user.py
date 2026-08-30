import os
import time
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
        
        for p in paragraphs:
            if any(term in p.lower() for term in search_terms):
                results.append(p)
                
        if not results:
            return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้"
        
        return "\n\n---\n\n".join(results)
    except FileNotFoundError:
        return "Error: ไม่พบไฟล์ knowledge_base.txt"

# 3. ตั้งค่า Google Gemini LLM (ใช้โมเดลที่รันผ่านในเครื่องของคุณ)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 4. Agent Node 1: Data Retriever
def data_retriever_node(state: AgentState):
    query = state["query"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Data Retriever. Extract 1-3 core keywords from the user's query to search a text database. Return ONLY the keywords separated by spaces. Do not translate."),
        ("user", "{query}")
    ])
    keyword_chain = prompt | llm
    keywords = keyword_chain.invoke({"query": query}).content.strip()
    retrieved_context = search_knowledge_base.invoke(keywords)
    return {"search_keywords": keywords, "context": retrieved_context}

# 5. Agent Node 2: Report Generator
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

# 6. ประกอบ Graph ด้วย LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("data_retriever", data_retriever_node)
workflow.add_node("report_generator", report_generator_node)
workflow.set_entry_point("data_retriever")
workflow.add_edge("data_retriever", "report_generator")
workflow.add_edge("report_generator", END)
app = workflow.compile()

# 7. ระบบรับ Input จากผู้ใช้แบบโต้ตอบ (Interactive Loop)
if __name__ == "__main__":
    print("==================================================")
    print("  ระบบ Agentic AI พร้อมใช้งาน (พิมพ์ 'exit' เพื่อออก)  ")
    print("==================================================\n")
    
    while True:
        # รับข้อความจากผู้ใช้งาน
        user_query = input("💬 ใส่คำถามของคุณ (User Query): ")
        
        # ตรวจสอบคำสั่งออก
        if user_query.strip().lower() in ['exit', 'quit', 'ออก']:
            print("👋 ปิดระบบ Agentic AI")
            break
            
        # ข้ามหากไม่ได้พิมพ์อะไรแล้วกด Enter
        if not user_query.strip():
            continue
            
        try:
            # ส่งคำถามเข้าระบบ Agent
            result = app.invoke({"query": user_query})
            print(f"\n🔍 Keywords Searched: [{result['search_keywords']}]")
            print(f"📝 Report Generator Output:\n{result['final_answer']}")
            
            # หน่วงเวลา 40 วินาทีหลังตอบสำเร็จ เพื่อเตรียมโควตาให้คำถามถัดไป
            print("\n⏳ (รอ 40 วินาทีเพื่อป้องกันโควตา API เต็ม...)")
            time.sleep(40)
            
        except Exception as e:
            # ดักจับ Error 429 และบังคับให้ระบบพักการทำงานอัตโนมัติ
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(f"\n❌ ระบบตรวจพบว่าโควตา API เต็มชั่วคราว (Rate Limit)")
                print("⏳ ระบบกำลังหยุดพัก 45 วินาทีโดยอัตโนมัติเพื่อรีเซ็ตโควตา...")
                time.sleep(45)
                print("✅ พร้อมใช้งานต่อแล้วครับ คุณสามารถพิมพ์คำถามเดิมซ้ำได้เลย")
            else:
                print(f"\n❌ เกิดข้อผิดพลาดจากระบบ: {e}")
                print("กรุณารอสักครู่แล้วลองถามใหม่อีกครั้ง")
            
        print("\n" + "-" * 60 + "\n")