

import os
import io
import tempfile
import requests
from datetime import datetime
import streamlit as st
from typing import Literal
from dotenv import load_dotenv
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from gtts import gTTS
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# Load env & init model
# ============================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize LLM (adjust model string if needed)
llm = init_chat_model("groq:llama-3.1-8b-instant")

# ============================================================
# State definition for LangGraph workflow
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
# Helper: Research / APIs
# ============================================================
def fetch_pubmed(topic: str, max_results=5):
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": topic, "retmax": max_results, "retmode": "json", "sort": "pub+date"}
        res = requests.get(url, params=params, timeout=15).json()
        ids = res.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed papers found."
        ids_str = ",".join(ids)
        sdata = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ids_str, "retmode": "json"},
            timeout=15
        ).json()
        summaries = []
        for pid in ids:
            doc = sdata.get("result", {}).get(pid, {})
            title = doc.get("title", "No title")
            authors = ", ".join([a.get("name","") for a in doc.get("authors", []) if isinstance(a, dict)])
            year = doc.get("pubdate", "Unknown").split(" ")[0]
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            summaries.append(f"{title} ({year})\nAuthors: {authors}\n{link}")
        return "\n\n".join(summaries)
    except Exception as e:
        return f"Error fetching PubMed: {e}"

def fetch_semantic_scholar(topic: str, max_results=5):
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": topic, "limit": max_results, "fields": "title,abstract,year,url"}
        res = requests.get(url, params=params, timeout=15).json()
        papers = res.get("data", [])
        if not papers:
            return "No Semantic Scholar papers found."
        return "\n\n".join([
            f"{p.get('title','')} ({p.get('year','')})\n{p.get('abstract','No abstract available')}\n{p.get('url','')}"
            for p in papers
        ])
    except Exception as e:
        return f"Error fetching Semantic Scholar: {e}"

def fetch_openalex(topic: str, max_results=5):
    try:
        url = "https://api.openalex.org/works"
        params = {"search": topic, "per-page": max_results}
        res = requests.get(url, params=params, timeout=15).json()
        results = res.get("results", [])
        if not results:
            return "No OpenAlex papers found."
        return "\n\n".join([f"{r.get('display_name','')} ({r.get('publication_year','')})\n{r.get('id','')}" for r in results])
    except Exception as e:
        return f"Error fetching OpenAlex: {e}"

def fetch_clinical_trials(topic: str, max_results=5):
    trials = []
    # ClinicalTrials.gov
    try:
        url = "https://clinicaltrials.gov/api/query/study_fields"
        params = {
            "expr": topic,
            "fields": "BriefTitle,Phase,OverallStatus,LocationCity,LocationCountry,StudyURL",
            "min_rnk": 1,
            "max_rnk": max_results,
            "fmt": "json"
        }
        res = requests.get(url, params=params, timeout=15).json()
        studies = res.get("StudyFieldsResponse", {}).get("StudyFields", [])
        if studies:
            ctgov = [
                f"{s.get('BriefTitle',[''])[0]} | {s.get('Phase',[''])[0]} | {s.get('OverallStatus',[''])[0]} | {s.get('LocationCity',[''])[0]}, {s.get('LocationCountry',[''])[0]}\n{s.get('StudyURL',[''])[0]}"
                for s in studies
            ]
            trials.append("=== ClinicalTrials.gov ===\n" + "\n".join(ctgov))
        else:
            trials.append("=== ClinicalTrials.gov ===\nNo trials found.")
    except Exception as e:
        trials.append(f"=== ClinicalTrials.gov ===\nError: {e}")
    # WHO ICTRP
    try:
        ictrp_url = f"https://trialsearch.who.int/TrialSearchAPI/search?query={topic}&page=1&pageSize={max_results}"
        res = requests.get(ictrp_url, timeout=15).json()
        records = res.get("records", [])
        if records:
            ictrp = [f"{r.get('scientificTitle','')} | Status: {r.get('recruitmentStatus','')} | URL: {r.get('url','')}" for r in records]
            trials.append("=== WHO ICTRP ===\n" + "\n".join(ictrp))
        else:
            trials.append("=== WHO ICTRP ===\nNo trials found.")
    except Exception as e:
        trials.append(f"=== WHO ICTRP ===\nError: {e}")
    # NCI
    try:
        nci_url = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"
        params = {"disease": topic, "size": max_results}
        res = requests.get(nci_url, params=params, timeout=15).json()
        trials_data = res.get("trials", [])
        if trials_data:
            nci = [f"{t.get('title','')} | Status: {t.get('status','')} | Phase: {t.get('phase','')} | URL: {t.get('url','')}" for t in trials_data]
            trials.append("=== NCI Cancer Trials ===\n" + "\n".join(nci))
        else:
            trials.append("=== NCI Cancer Trials ===\nNo trials found.")
    except Exception as e:
        trials.append(f"=== NCI Cancer Trials ===\nError: {e}")
    return "\n\n".join(trials)

# ============================================================
# Agents for multi-agent research workflow
# ============================================================
def supervisor_agent(state: MedState):
    has_research = bool(state.get("research_data", ""))
    has_drug_discovery = bool(state.get("drug_discovery", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))
    if not has_research:
        next_agent = "researcher"
        msg = "🔬 Assigning task to Researcher..."
    elif not has_drug_discovery:
        next_agent = "drug_discovery"
        msg = "💊 Assigning task to Drug Discovery Agent..."
    elif not has_analysis:
        next_agent = "analyst"
        msg = "📊 Assigning task to Analyst..."
    elif not has_report:
        next_agent = "writer"
        msg = "✍️ Assigning task to Writer..."
    else:
        next_agent = "end"
        msg = "✅ All tasks complete!"
    return {"messages": [AIMessage(content=msg)], "next_agent": next_agent}

def researcher_agent(state: dict) -> dict:
    topic = state.get("current_task", "")
    pubmed_data = fetch_pubmed(topic)
    sem_data = fetch_semantic_scholar(topic)
    openalex_data = fetch_openalex(topic)
    research_text = f"""
=== PUBMED RESULTS ===
{pubmed_data}

=== SEMANTIC SCHOLAR RESULTS ===
{sem_data}

=== OPENALEX RESULTS ===
{openalex_data}
"""
    msg = f"🧠 Researcher Agent:\nCompiled verified medical papers for **{topic}**."
    return {"messages": [AIMessage(content=msg)], "research_data": research_text, "next_agent": "supervisor"}

def drug_discovery_agent(state: MedState):
    topic = state.get("current_task", "")
    search_query = f"{topic} new drug OR therapeutic OR treatment discovery"
    pubmed_drugs = fetch_pubmed(search_query)
    semantic_drugs = fetch_semantic_scholar(search_query)
    openalex_drugs = fetch_openalex(search_query)
    clinical_trials = fetch_clinical_trials(topic)
    combined_drug_data = f"""
=== PUBMED DRUG DISCOVERY ===
{pubmed_drugs}

=== SEMANTIC SCHOLAR DRUG DISCOVERY ===
{semantic_drugs}

=== OPENALEX DRUG DISCOVERY ===
{openalex_drugs}

=== ONGOING CLINICAL TRIALS ===
{clinical_trials}
"""
    prompt = f"""
You are a pharmacology research expert.
Summarize:
1. New or experimental drugs related to '{topic}'
2. Mechanisms of action
3. Clinical trial outcomes & ongoing studies
4. Future drug development directions
5. Key pharmaceutical companies or research groups

Data:
{combined_drug_data[:3000]}
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
Analyze the combined research and drug data on '{topic}'.
Provide:
1. Major findings and evidence strength
2. Key trends across studies
3. Gaps and limitations
4. Future research directions
Data:
{research_data[:1500]}
Drug Discovery:
{drug_data[:1500]}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content
    msg = f"📊 Analyst:\n{analysis[:800]}"
    return {"messages": [AIMessage(content=msg)], "analysis": analysis, "next_agent": "supervisor"}

def writer_agent(state: MedState):
    topic = state.get("current_task")
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    drug_data = state.get("drug_discovery", "")
    prompt = f"""Write a comprehensive medical literature review on '{topic}'.
Sections:
1. Executive Summary
2. Key Findings
3. New Drug Discoveries & Ongoing Trials
4. Trends & Gaps
5. Recommendations for clinicians/researchers
6. References
Base your report on:
Research Data: {research_data[:1000]}
Drug Data: {drug_data[:1000]}
Analysis: {analysis[:1000]}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content
    final_report = f"""
🧾 MEDICAL LITERATURE REVIEW
==================================================
Topic: {topic}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
==================================================
{report}
==================================================
"""
    return {"messages": [AIMessage(content="✍️ Writer: Final report completed!")], "final_report": final_report, "task_complete": True, "next_agent": "supervisor"}

# Router + workflow compile
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
    workflow.add_conditional_edges(node, router, {
        "supervisor": "supervisor",
        "researcher": "researcher",
        "drug_discovery": "drug_discovery",
        "analyst": "analyst",
        "writer": "writer",
        END: END
    })
graph = workflow.compile(checkpointer=MemorySaver())

# ============================================================
# Utilities: PDF, TTS, Knowledge graph, timeline, podcast
# ============================================================
def generate_pdf(topic, messages, final_report):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = [Paragraph(f"<b>Medical Research Review</b>", styles['Title']), Spacer(1, 12),
               Paragraph(f"<b>Topic:</b> {topic}", styles['Heading2']), Spacer(1, 12)]
    for m in messages:
        content.append(Paragraph(m.replace("\n", "<br/>"), styles['BodyText']))
        content.append(Spacer(1, 8))
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Final Report:</b>", styles['Heading2']))
    content.append(Paragraph(final_report.replace("\n", "<br/>"), styles['BodyText']))
    doc.build(content)
    buffer.seek(0)
    return buffer

def speak_text_and_get_audio_bytes(text, lang="en"):
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

def generate_medical_timeline(topic):
    prompt = f"""
Create a structured timeline of major milestones for the disease: {topic}.
Format as lines: YEAR - EVENT
Start from earliest known discovery to the latest breakthroughs.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_knowledge_graph(topic):
    prompt = f"""
Provide several relationships for a medical knowledge graph for: {topic}
Format lines like:
Disease -> Symptom -> Drug -> Organization -> Country
One relation per line.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    G = nx.Graph()
    for line in response.content.splitlines():
        if "->" in line:
            nodes = [n.strip() for n in line.split("->") if n.strip()]
            for i in range(len(nodes)-1):
                G.add_edge(nodes[i], nodes[i+1])
    return G, response.content

def generate_podcast_script(topic, report_text):
    prompt = f"""
Convert the following medical report into a 3-minute, engaging podcast script aimed at a general audience.
Topic: {topic}
Report:
{report_text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_drug_match(patient_profile, topic):
    prompt = f"""
You are a clinical research assistant. Based on this patient profile:
{patient_profile}
and disease/topic: {topic}
Provide:
- Suggested treatments (with short rationale)
- Relevant ongoing clinical trials (IDs or sources if available)
- Main precautions/contraindications
Return as a structured list.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="BioVerse AI", layout="wide", page_icon="🧬")
st.title("🧬 BioVerse AI")
st.write("Automated medical literature workflow + creative tools (timeline, matchmaking, knowledge graph, podcast, chat & voice).")
st.sidebar.title("Settings & Modes")

# Sidebar controls
mode = st.sidebar.radio("App Mode", ["Research Mode", "Chat Assistant", "Doctor Assistant Mode"])
personality_options = {
    "Calm Doctor 🩺": "You are a calm, concise medical doctor. Provide evidence-based, clear clinical explanations. If unsure, say so and recommend consulting a physician. Prioritize patient safety.",
    "Research Scientist 🔬": "You are a medical research scientist. Provide thorough, citation-aware summaries and study-level evidence strength. Use technical language for clinicians/researchers.",
    "Compassionate Clinician ❤️": "You are a compassionate clinician who explains medical information in simple, empathetic language. Reassure the user and provide clear next steps.",
    "Professor 🎓": "You are a medical professor. Explain step-by-step, give examples and exam-style questions."
}
personality = st.sidebar.selectbox("Assistant Personality", list(personality_options.keys()))
attach_report_context = st.sidebar.checkbox("Attach latest report excerpt to context", value=True)
tts_lang = st.sidebar.selectbox("TTS Language", ["en", "en-in"], index=0)

# Main layout columns: Left - research controls; Right - chat / assistant
left_col, right_col = st.columns([1.6, 1])

with left_col:
    st.header("🔬 Literature Research & Reports")
    topic = st.text_input("Enter medical topic (e.g., 'Pneumonia treatment')", placeholder="e.g., Lung Cancer treatment")
    st.markdown("**Quick features:** timeline, drug matchmaking, knowledge graph, podcast, PDF export.")

    # Start Research (multi-agent)
    if st.button("Start Research") and topic.strip():
        state = MedState(messages=[HumanMessage(content=topic)], current_task=topic)
        st.info("🚀 Running automated literature review pipeline...")
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
                    
                    # Save to MySQL using our drug.py helpers
                    try:
                        import drug
                        # Assuming user_id=None for anonymous or guest user
                        drug.save_query_to_mysql(user_id=None, topic=topic, blood_group="N/A", condition="N/A", result_summary=value["final_report"][:2000])
                        drug.save_report_to_mysql(user_id=None, topic=topic, pdf_bytes=pdf_buffer.getvalue())
                        st.success("💾 Saved research to database successfully!")
                    except Exception as e:
                        st.warning(f"Could not save to database: {e}")

                    st.download_button(label="📥 Download Medical Report (PDF)",
                                       data=pdf_buffer,
                                       file_name=f"{topic.replace(' ', '_')}_Medical_Report.pdf",
                                       mime="application/pdf")
        st.success("✅ Research pipeline completed. Use the right pane for chat, timeline, podcast and more.")

    # Timeline
    st.markdown("---")
    st.header("📅 Medical Timeline")
    if st.button("Generate Medical Timeline") and topic.strip():
        if not topic:
            st.warning("Please enter a topic above.")
        else:
            st.info("Generating timeline...")
            timeline_text = generate_medical_timeline(topic)
            st.text_area("Medical Timeline (Year - Event)", timeline_text, height=250)

    # Drug Matchmaking / Patient Profile
    st.markdown("---")
    st.header("🧑‍⚕️ AI Drug Matchmaking (Patient Profile)")
    patient_profile = st.text_area("Enter patient profile (age, comorbidities, symptoms):",
                                   placeholder="65 yo, diabetic, hypertension, stage II lung cancer...")
    if st.button("Find Suitable Treatments"):
        if not patient_profile.strip() or not topic.strip():
            st.warning("Provide both a topic and a patient profile.")
        else:
            st.info("Finding best-suited treatments...")
            match_out = generate_drug_match(patient_profile, topic)
            st.markdown("### 💊 Treatment Match Output")
            st.write(match_out)
            st.warning("⚠️ For educational/research use only. Not medical advice.")

    # Knowledge Graph
    st.markdown("---")
    st.header("🧠 Knowledge Graph")
    if st.button("Generate Knowledge Graph") and topic.strip():
        st.info("Creating knowledge graph from LLM relations...")
        G, raw_rel = generate_knowledge_graph(topic)
        if len(G.nodes) == 0:
            st.info("No relations found or LLM didn't return relations.")
            st.text_area("Raw Relations", raw_rel, height=150)
        else:
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=1600, font_size=9)
            st.pyplot(plt)
            st.text_area("Raw Relations (LLM)", raw_rel, height=150)

    # Podcast generator
    st.markdown("---")
    st.header("🎙️ AI Medical Podcast Generator")
    if st.button("Generate Podcast Script") and topic.strip():
        report_text = st.session_state.get("final_report", "")
        if not report_text:
            st.warning("Generate the research report first (Start Research) before creating a podcast.")
        else:
            st.info("Generating podcast script...")
            script = generate_podcast_script(topic, report_text)
            st.text_area("Podcast Script (editable)", script, height=300)
            # TTS preview
            audio_bytes = speak_text_and_get_audio_bytes(script, lang=tts_lang)
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("⬇️ Download Podcast (MP3)", data=audio_bytes, file_name="podcast.mp3", mime="audio/mp3")

with right_col:
    st.header("💬 Assistant & Doctor Modes")
    st.info("⚠️ This assistant is for educational/research purposes only and is NOT medical advice.")

    # initialize chat history and last audio
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_tts_audio" not in st.session_state:
        st.session_state.last_tts_audio = None

    # Chat Assistant Mode
    if mode == "Chat Assistant":
        st.subheader("Chat Assistant")
        user_q = st.text_input("Ask a question about the topic/report:")
        if st.button("Ask") and user_q.strip():
            st.session_state.chat_history.append(("user", user_q))
            system_prompt = SystemMessage(content=personality_options[personality])
            messages = [system_prompt]
            if attach_report_context and st.session_state.get("final_report"):
                messages.append(HumanMessage(content=f"Context (report excerpt): {st.session_state['final_report'][:2000]}"))
            # append conversation
            for role, text in st.session_state.chat_history:
                if role == "user":
                    messages.append(HumanMessage(content=text))
                else:
                    messages.append(AIMessage(content=text))
            response = llm.invoke(messages)
            assistant_text = response.content
            st.session_state.chat_history.append(("assistant", assistant_text))
            st.markdown(f"**Assistant ({personality}):**")
            st.write(assistant_text)
            st.session_state.last_tts_audio = speak_text_and_get_audio_bytes(assistant_text, lang=tts_lang)

        # show conversation
        if st.session_state.chat_history:
            st.markdown("### Conversation")
            for role, text in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"**You:** {text}")
                else:
                    st.markdown(f"**Assistant ({personality}):** {text}")

        # TTS controls
        if st.session_state.get("last_tts_audio"):
            if st.button("🔊 Play Last Assistant Response"):
                st.audio(st.session_state["last_tts_audio"], format="audio/mp3")
            st.download_button("⬇️ Download Last Assistant Audio", data=st.session_state["last_tts_audio"],
                               file_name="assistant_response.mp3", mime="audio/mp3")

    # Doctor Assistant Mode (voice-enabled)
    elif mode == "Doctor Assistant Mode":
        st.subheader("Doctor Assistant (Voice-enabled)")
        st.markdown("Enter patient profile and question or upload a short audio (phone-recorded) to transcribe.")
        patient_profile_doctor = st.text_area("Patient profile (brief):", placeholder="e.g., 67yo male, COPD, diabetes")
        doctor_question = st.text_input("Question for doctor (or leave blank to use uploaded audio):")

        uploaded_audio = st.file_uploader("Upload voice question (mp3/wav) — optional", type=["mp3", "wav", "m4a"])
        if uploaded_audio is not None:
            st.info("Uploaded audio. Transcription requires a transcription backend (Whisper or other).")
            if st.button("Transcribe & Use uploaded audio"):
                audio_bytes = uploaded_audio.read()
                # Placeholder: integrate your chosen STT here (Whisper or cloud STT)
                # Example: transcription = transcribe_with_whisper(audio_bytes)
                transcription = "TRANSCRIPTION_PLACEHOLDER: (replace with real transcription output)"
                st.markdown(f"**Transcribed text:** {transcription}")
                doctor_question = transcription
                st.session_state.chat_history.append(("user", transcription))

        if st.button("Ask Doctor (Text)"):
            system_doctor = SystemMessage(content="""
You are a cautious, evidence-first physician. For each user query, return:
1) A single 'URGENCY: Low/Moderate/High' tag on the first line
2) A concise clinical answer with suggested next steps
3) When to seek emergency care
4) If uncertain, clearly say so and recommend in-person evaluation
""")
            combined = f"Patient profile: {patient_profile_doctor}\nQuestion: {doctor_question}\nTopic: {topic or 'N/A'}"
            messages = [system_doctor]
            if attach_report_context and st.session_state.get("final_report"):
                messages.append(HumanMessage(content=f"Context (report excerpt): {st.session_state['final_report'][:1500]}"))
            messages.append(HumanMessage(content=combined))
            response = llm.invoke(messages)
            assistant_text = response.content
            st.session_state.chat_history.append(("assistant", assistant_text))

            # parse urgency
            urgency = None
            first_line = assistant_text.splitlines()[0] if assistant_text else ""
            if first_line.upper().startswith("URGENCY:"):
                urgency = first_line.split(":", 1)[1].strip().lower()
            if urgency == "high":
                st.error("🔴 URGENCY: HIGH — Immediate medical attention recommended.")
            elif urgency == "moderate":
                st.warning("🟠 URGENCY: MODERATE — Prompt clinical review advised.")
            elif urgency == "low":
                st.success("🟢 URGENCY: LOW — Non-urgent; follow suggested steps.")
            else:
                st.info("⚪ URGENCY: Not detected in assistant output.")

            st.markdown("**Doctor Assistant Response:**")
            st.write(assistant_text)
            st.session_state.last_tts_audio = speak_text_and_get_audio_bytes(assistant_text, lang=tts_lang)

        # TTS play / download
        if st.session_state.get("last_tts_audio"):
            if st.button("🔊 Play Doctor Assistant Response"):
                st.audio(st.session_state["last_tts_audio"], format="audio/mp3")
            st.download_button("⬇️ Download Doctor Assistant Audio", data=st.session_state["last_tts_audio"],
                               file_name="doctor_response.mp3", mime="audio/mp3")

    # Research Mode / Quick utilities
    else:
        st.subheader("Research Mode - Quick Tools")
        st.markdown("- Use **Start Research** on the left to create a full report.")
        if st.button("Generate Quick Timeline (LLM)"):
            if not topic:
                st.warning("Enter a topic first.")
            else:
                timeline = generate_medical_timeline(topic)
                st.text_area("Timeline", timeline, height=250)
        if st.button("Generate Quick Knowledge Graph"):
            if not topic:
                st.warning("Enter a topic first.")
            else:
                G, raw_rel = generate_knowledge_graph(topic)
                if len(G.nodes) == 0:
                    st.text_area("Raw Relations", raw_rel, height=150)
                else:
                    plt.figure(figsize=(8,5))
                    nx.draw(G, with_labels=True, node_size=1400, font_size=9)
                    st.pyplot(plt)


