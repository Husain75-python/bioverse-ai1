import os
import io
import re
import tempfile
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from gtts import gTTS

import networkx as nx
import matplotlib.pyplot as plt

# Try importing MySQL helpers
try:
    import drug
except ImportError:
    drug = None

# ============================================================
# Environment & LLM Config
# ============================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)

# ============================================================
# LangGraph State
# ============================================================
class MedState(MessagesState):
    next_agent: str = ""
    research_data: str = ""
    drug_discovery: str = ""
    analysis: str = ""
    final_report: str = ""
    current_task: str = ""
    task_complete: bool = False

# ============================================================
# API Helpers (Optimized with ThreadPoolExecutor)
# ============================================================
def fetch_pubmed(topic: str, max_results=5) -> str:
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": topic, "retmax": max_results, "retmode": "json", "sort": "pub+date"}
        res = requests.get(url, params=params, timeout=10).json()
        ids = res.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed papers found."
        
        ids_str = ",".join(ids)
        sdata = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ids_str, "retmode": "json"},
            timeout=10
        ).json()
        
        summaries = []
        for pid in ids:
            doc = sdata.get("result", {}).get(pid, {})
            title = doc.get("title", "No title")
            authors = ", ".join([a.get("name", "") for a in doc.get("authors", []) if isinstance(a, dict)])
            year = doc.get("pubdate", "Unknown").split(" ")[0]
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            summaries.append(f"• {title} ({year})\nAuthors: {authors}\nURL: {link}")
        return "\n\n".join(summaries)
    except Exception as e:
        return f"Error fetching PubMed: {e}"

def fetch_semantic_scholar(topic: str, max_results=5) -> str:
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": topic, "limit": max_results, "fields": "title,abstract,year,url"}
        res = requests.get(url, params=params, timeout=10).json()
        papers = res.get("data", [])
        if not papers:
            return "No Semantic Scholar papers found."
        return "\n\n".join([
            f"• {p.get('title','') if p.get('title') else 'Untitled'} ({p.get('year','')})\n{p.get('abstract','No abstract available')}\nURL: {p.get('url','')}"
            for p in papers
        ])
    except Exception as e:
        return f"Error fetching Semantic Scholar: {e}"

def fetch_openalex(topic: str, max_results=5) -> str:
    try:
        url = "https://api.openalex.org/works"
        params = {"search": topic, "per-page": max_results}
        res = requests.get(url, params=params, timeout=10).json()
        results = res.get("results", [])
        if not results:
            return "No OpenAlex papers found."
        return "\n\n".join([f"• {r.get('display_name','')} ({r.get('publication_year','')})\nID: {r.get('id','')}" for r in results])
    except Exception as e:
        return f"Error fetching OpenAlex: {e}"

def fetch_clinical_trials(topic: str, max_results=5) -> str:
    try:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {"query.term": topic, "pageSize": max_results}
        res = requests.get(url, params=params, timeout=10).json()
        studies = res.get("studies", [])
        if not studies:
            return "No ClinicalTrials.gov trials found."
        
        parsed = []
        for s in studies:
            protocol = s.get("protocolSection", {})
            title = protocol.get("identificationModule", {}).get("briefTitle", "No Title")
            status = protocol.get("statusModule", {}).get("overallStatus", "Unknown")
            phases = ", ".join(protocol.get("designModule", {}).get("phases", ["N/A"]))
            parsed.append(f"• {title} | Status: {status} | Phase: {phases}")
        return "=== ClinicalTrials.gov ===\n" + "\n".join(parsed)
    except Exception as e:
        return f"ClinicalTrials.gov Fetch Error: {e}"

# Parallel Research Fetcher
def fetch_all_research_parallel(topic: str) -> dict:
    with ThreadPoolExecutor(max_workers=4) as executor:
        f_pubmed = executor.submit(fetch_pubmed, topic)
        f_semantic = executor.submit(fetch_semantic_scholar, topic)
        f_openalex = executor.submit(fetch_openalex, topic)
        f_trials = executor.submit(fetch_clinical_trials, topic)
        
        return {
            "pubmed": f_pubmed.result(),
            "semantic": f_semantic.result(),
            "openalex": f_openalex.result(),
            "trials": f_trials.result(),
        }

# ============================================================
# Multi-Agent Workflow
# ============================================================
def supervisor_agent(state: MedState):
    if not bool(state.get("research_data", "")):
        return {"messages": [AIMessage(content="🔬 Assigning task to Researcher...")], "next_agent": "researcher"}
    elif not bool(state.get("drug_discovery", "")):
        return {"messages": [AIMessage(content="💊 Assigning task to Drug Discovery Agent...")], "next_agent": "drug_discovery"}
    elif not bool(state.get("analysis", "")):
        return {"messages": [AIMessage(content="📊 Assigning task to Analyst...")], "next_agent": "analyst"}
    elif not bool(state.get("final_report", "")):
        return {"messages": [AIMessage(content="✍️ Assigning task to Writer...")], "next_agent": "writer"}
    else:
        return {"messages": [AIMessage(content="✅ All tasks complete!")], "next_agent": "end"}

def researcher_agent(state: dict) -> dict:
    topic = state.get("current_task", "")
    data = fetch_all_research_parallel(topic)
    research_text = f"""
=== PUBMED RESULTS ===
{data['pubmed']}

=== SEMANTIC SCHOLAR RESULTS ===
{data['semantic']}

=== OPENALEX RESULTS ===
{data['openalex']}
"""
    msg = f"🧠 Researcher Agent:\nCompiled verified medical papers for **{topic}**."
    return {"messages": [AIMessage(content=msg)], "research_data": research_text, "next_agent": "supervisor"}

def drug_discovery_agent(state: MedState):
    topic = state.get("current_task", "")
    data = fetch_all_research_parallel(f"{topic} novel therapeutic drug discovery")
    
    combined_drug_data = f"""
=== PUBMED & SCHOLAR DISCOVERY ===
{data['pubmed']}
{data['semantic']}

=== CLINICAL TRIALS ===
{data['trials']}
"""
    prompt = f"""You are a pharmacology research expert. Summarize:
1. New or experimental drugs related to '{topic}'
2. Mechanisms of action
3. Ongoing clinical trials
4. Future therapeutic directions

Data:
{combined_drug_data}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    drug_summary = response.content
    msg = f"💊 Drug Discovery Agent:\n{drug_summary[:800]}"
    return {"messages": [AIMessage(content=msg)], "drug_discovery": drug_summary, "next_agent": "supervisor"}

def analyst_agent(state: MedState):
    topic = state.get("current_task")
    research_data = state.get("research_data", "")
    drug_data = state.get("drug_discovery", "")
    
    prompt = f"""You are a biomedical research analyst.
Analyze the provided research and drug data on '{topic}'.
Provide:
1. Major findings and evidence strength
2. Trends across studies
3. Research gaps and future directions

Data:
{research_data}
{drug_data}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content
    msg = f"📊 Analyst:\n{analysis[:800]}"
    return {"messages": [AIMessage(content=msg)], "analysis": analysis, "next_agent": "supervisor"}

def writer_agent(state: MedState):
    topic = state.get("current_task")
    research_data = state.get("research_data", "")[:3000]
    analysis = state.get("analysis", "")[:2000]
    drug_data = state.get("drug_discovery", "")[:2000]    
    prompt = f"""Write a comprehensive medical literature review on '{topic}'.
Structure:
1. Executive Summary
2. Key Findings
3. New Drug Discoveries & Clinical Trials
4. Research Trends & Gaps
5. Clinical Recommendations

Base your review on:
Research: {research_data}
Drug Data: {drug_data}
Analysis: {analysis}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content
    final_report = f"🧾 MEDICAL LITERATURE REVIEW\nTopic: {topic}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{report}"
    
    return {
        "messages": [AIMessage(content="✍️ Writer: Final report completed!")],
        "final_report": final_report,
        "task_complete": True,
        "next_agent": "supervisor"
    }

def router(state: MedState) -> Literal["supervisor", "researcher", "drug_discovery", "analyst", "writer", "END"]:
    if state.get("task_complete", False):
        return END
    return state.get("next_agent", "supervisor")

workflow = StateGraph(MedState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("drug_discovery", drug_discovery_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("supervisor")

for node in ["supervisor", "researcher", "drug_discovery", "analyst", "writer"]:
    workflow.add_conditional_edges(
        node, router,
        {
            "supervisor": "supervisor",
            "researcher": "researcher",
            "drug_discovery": "drug_discovery",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        }
    )
graph = workflow.compile(checkpointer=MemorySaver())

# ============================================================
# Utilities: Sanitized PDF Generator & Tools
# ============================================================
def clean_text_for_pdf(text: str) -> str:
    """Escapes XML and strips Markdown formatting for ReportLab compatibility."""
    if not text:
        return ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|[-:\s|]+\|", "", text)
    text = text.replace("|", "  ")
    return text

def generate_pdf(topic: str, messages: list, final_report: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    content = [
        Paragraph("<b>Medical Research Review</b>", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"<b>Topic:</b> {clean_text_for_pdf(topic)}", styles['Heading2']),
        Spacer(1, 12)
    ]
    
    for m in messages:
        cleaned_msg = clean_text_for_pdf(m)
        for line in cleaned_msg.split("\n"):
            if line.strip():
                content.append(Paragraph(line.strip(), styles['BodyText']))
                content.append(Spacer(1, 4))
    
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Final Report:</b>", styles['Heading2']))
    
    cleaned_report = clean_text_for_pdf(final_report)
    for line in cleaned_report.split("\n"):
        if line.strip():
            content.append(Paragraph(line.strip(), styles['BodyText']))
            content.append(Spacer(1, 4))
            
    doc.build(content)
    buffer.seek(0)
    return buffer

@st.cache_data
def speak_text_and_get_audio_bytes(text: str, lang="en") -> bytes:
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.close()
    tts.save(tmp_name)
    with open(tmp_name, "rb") as f:
        audio_bytes = f.read()
    try:
        os.remove(tmp_name)
    except Exception:
        pass
    return audio_bytes

def generate_medical_timeline(topic: str) -> str:
    prompt = f"Create a structured chronological timeline of major milestones for: {topic}. Format each line as: YEAR - EVENT."
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_knowledge_graph(topic: str):
    prompt = f"Provide medical relationships for {topic}. Return output line-by-line formatted as: NodeA -> NodeB"
    resp = llm.invoke([HumanMessage(content=prompt)]).content
    G = nx.Graph()
    for line in resp.splitlines():
        if "->" in line:
            nodes = [n.strip() for n in line.split("->") if n.strip()]
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i+1])
    return G, resp

def generate_podcast_script(topic: str, report_text: str) -> str:
    prompt = f"Convert this report into a 2-minute podcast script for a general audience.\nTopic: {topic}\nReport:\n{report_text}"
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_drug_match(patient_profile: str, topic: str) -> str:
    prompt = f"Patient Profile: {patient_profile}\nTopic: {topic}\nProvide suggested treatments, contraindications, and relevant trial types."
    return llm.invoke([HumanMessage(content=prompt)]).content

# ============================================================
# Streamlit Dashboard Interface
# ============================================================
st.set_page_config(page_title="BioVerse AI", layout="wide", page_icon="🧬")
st.title("🧬 BioVerse AI")
st.caption("Multi-Agent Medical Literature Review & AI Clinical Assistant")

st.sidebar.title("Configuration")
mode = st.sidebar.radio("Navigation", ["Research Mode", "Clinical Assistant"])
personality_options = {
    "Calm Doctor 🩺": "You are a calm, concise medical doctor. Prioritize evidence and patient safety.",
    "Research Scientist 🔬": "You are a research scientist focusing on methodology, citations, and evidence strength.",
    "Compassionate Clinician ❤️": "You explain complex medical terms in empathetic, lay-friendly language."
}
personality = st.sidebar.selectbox("Assistant Persona", list(personality_options.keys()))

left_col, right_col = st.columns([1.6, 1])

with left_col:
    st.header("🔬 Literature Research Engine")
    topic = st.text_input("Enter Medical Topic/Condition:", placeholder="e.g., Non-Small Cell Lung Cancer")
    
    if st.button("🚀 Run Multi-Agent Research") and topic.strip():
        state = MedState(messages=[HumanMessage(content=topic)], current_task=topic)
        st.info("Running pipeline...")
        full_messages = []
        config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}
        
        for event in graph.stream(state, config=config):
            for value in event.values():
                if "messages" in value:
                    for msg in value["messages"]:
                        full_messages.append(msg.content)
                        st.markdown(f"**{msg.content}**")
                if "final_report" in value and value["final_report"]:
                    st.markdown("### 🧾 Final Report")
                    st.code(value["final_report"])
                    st.session_state["final_report"] = value["final_report"]
                    
                    pdf_buffer = generate_pdf(topic, full_messages, value["final_report"])
                    
                    if drug:
                        user_id = st.session_state.get("user_id", None)
                        drug.save_query_to_mysql(user_id=user_id, topic=topic, blood_group="N/A", condition="N/A", result_summary=value["final_report"][:2000])
                        drug.save_report_to_mysql(user_id=user_id, topic=topic, pdf_bytes=pdf_buffer.getvalue())
                        st.success("💾 Saved report to database!")
                    
                    st.download_button(
                        label="📥 Download Medical Report (PDF)",
                        data=pdf_buffer,
                        file_name=f"{topic.replace(' ', '_')}_Report.pdf",
                        mime="application/pdf"
                    )

    st.markdown("---")
    st.subheader("📅 Disease Milestones Timeline")
    if st.button("Generate Timeline") and topic.strip():
        st.text_area("Timeline Output", generate_medical_timeline(topic), height=200)

    st.markdown("---")
    st.subheader("🧑‍⚕️ Patient Drug Matchmaking")
    patient_profile = st.text_area("Patient Profile (Age, Symptoms, Co-morbidities):")
    if st.button("Find Treatments") and patient_profile.strip():
        st.write(generate_drug_match(patient_profile, topic))

    st.markdown("---")
    st.subheader("🧠 Knowledge Graph")
    if st.button("Generate Visual Graph") and topic.strip():
        G, raw = generate_knowledge_graph(topic)
        if len(G.nodes) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=1200, font_size=8, ax=ax)
            st.pyplot(fig)
            plt.clf()
        else:
            st.warning("No relationships extracted.")

    st.markdown("---")
    st.subheader("🎙️ AI Medical Podcast")
    if st.button("Generate Audio Summary") and st.session_state.get("final_report"):
        script = generate_podcast_script(topic, st.session_state["final_report"])
        st.text_area("Podcast Script", script, height=200)
        audio_data = speak_text_and_get_audio_bytes(script)
        st.audio(audio_data, format="audio/mp3")

with right_col:
    st.header("💬 AI Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    user_q = st.text_input("Ask a clinical or research question:")
    if st.button("Send") and user_q.strip():
        st.session_state.chat_history.append(("user", user_q))
        
        sys_msg = SystemMessage(content=personality_options[personality])
        messages = [sys_msg]
        
        if st.session_state.get("final_report"):
            messages.append(HumanMessage(content=f"Context: {st.session_state['final_report'][:1500]}"))
            
        for role, txt in st.session_state.chat_history:
            messages.append(HumanMessage(content=txt) if role == "user" else AIMessage(content=txt))
            
        resp = llm.invoke(messages).content
        st.session_state.chat_history.append(("assistant", resp))
    
    for role, text in st.session_state.chat_history:
        st.markdown(f"**{'You' if role=='user' else 'Assistant'}:** {text}")
