# app.py
# ------------------------------------------------------------------
# Real-Estate RAG Chatbot: Pinecone + LangChain + LangGraph + TTS
# ------------------------------------------------------------------

import io, os, re, json
from typing import Dict, Any, List, TypedDict
from datetime import datetime
from dotenv import load_dotenv

import torch
import pandas as pd
import soundfile as sf
import streamlit as st

from transformers import VitsModel, AutoTokenizer

from langgraph.graph import StateGraph, END

from vector_store import RealEstateVectorStore  # <-- module mới

# -------------------- UI basics --------------------
st.set_page_config(page_title="AI tìm nhà (RAG + Web + TTS)", page_icon="🏠", layout="wide")
st.title("🏠 Chatbot Tìm Kiếm Bất Động Sản (Pinecone + LangChain + LangGraph)")
st.caption("Pinecone • LangChain • LangGraph • Azure OpenAI (tuỳ chọn) • Tavily Search • VITS TTS")
load_dotenv()

# -------------------- Optional: Tavily Web Search --------------------
TAVILY_ENABLED = bool(os.getenv("TAVILY_API_KEY"))
try:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if TAVILY_ENABLED else None
except Exception:
    tavily_client = None
    TAVILY_ENABLED = False

# -------------------- TTS helpers --------------------
@st.cache_resource(show_spinner=True)
def load_vits(model_name: str = "facebook/mms-tts-vie"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device).eval()
    sr = int(getattr(model.config, "sampling_rate", 16000))
    return model, tokenizer, device, sr

@torch.no_grad()
def tts_generate(model: VitsModel, tokenizer: AutoTokenizer, device: str, text: str):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    wav = model(**inputs).waveform.squeeze().detach().cpu()
    return wav

def wav_bytes_from_tensor(waveform: torch.Tensor, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, waveform.numpy(), sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return buf.read()

# -------------------- CSV parse helpers --------------------
PRICE_PAT = re.compile(r"([0-9]+(?:[\.,][0-9]+)?)\s*(tỷ|ty|tr|triệu)?", re.IGNORECASE)

def parse_price_to_billion(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    m = PRICE_PAT.search(s)
    if not m:
        return float("nan")
    num = m.group(1).replace(".", "").replace(",", ".")
    unit = (m.group(2) or "tỷ").lower()
    v = float(num)
    if unit in ["tr", "triệu"]:
        return v / 1000.0
    return v  # tỷ

def to_int_safe(x):
    try:
        return int(re.findall(r"\d+", str(x))[0])
    except Exception:
        return None

# -------------------- Sidebar settings --------------------
with st.sidebar:
    st.header("⚙️ Cấu hình")
    data_src = st.text_input("CSV path", value="real_estate_listings.csv")

    embed_model_name = st.text_input(
        "Embedding model (HuggingFace)",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )
    vits_model_name = st.text_input("VITS model", value="facebook/mms-tts-vie")

    st.markdown("**Pinecone (Vector DB)**")
    pinecone_api_key = st.text_input(
        "PINECONE_API_KEY",
        value=os.getenv("PINECONE_API_KEY", ""),
        type="password",
    )
    pinecone_index_name = st.text_input(
        "Pinecone index name",
        value=os.getenv("PINECONE_INDEX_NAME", "real-estate-rag"),
    )
    pinecone_cloud = st.text_input("Pinecone cloud", value="aws")
    pinecone_region = st.text_input("Pinecone region", value="us-east-1")

    st.markdown("**Azure OpenAI (tùy chọn)**")
    aoai_ep = st.text_input("AZURE_OPENAI_ENDPOINT", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    aoai_key = st.text_input("AZURE_OPENAI_API_KEY", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
    aoai_dep = st.text_input("AZURE_OPENAI_DEPLOYMENT", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))

    btn_rebuild = st.button("🔁 Rebuild index (Pinecone)")

# -------------------- Load CSV --------------------
@st.cache_data(show_spinner=True)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Price" in df.columns:
        df["price_billion"] = df["Price"].apply(parse_price_to_billion)
    for col, newc in [("Bedrooms","bedrooms"),("Toilets","toilets"),("Total Floors","floors")]:
        if col in df.columns:
            df[newc] = df[col].apply(to_int_safe)
    df["doc_text"] = df.apply(lambda r: " | ".join([str(v) for v in r.values]), axis=1).str.lower()
    return df

try:
    df = load_df(data_src)
    st.success(f"Đã nạp {len(df)} dòng từ ‘{data_src}’.")
    st.dataframe(df.head(10), width="stretch")
except Exception as e:
    st.error(f"Không đọc được CSV: {e}")
    st.stop()

# -------------------- Vector store (Pinecone + LangChain) --------------------
if not pinecone_api_key:
    st.error("Vui lòng cung cấp PINECONE_API_KEY trong sidebar hoặc biến môi trường.")
    st.stop()

@st.cache_resource(show_spinner=True)
def get_vector_store(
    api_key: str,
    index_name: str,
    cloud: str,
    region: str,
    embed_model_name: str,
) -> RealEstateVectorStore:
    return RealEstateVectorStore(
        api_key=api_key,
        index_name=index_name,
        cloud=cloud,
        region=region,
        embed_model_name=embed_model_name,
    )

vector_store = get_vector_store(
    pinecone_api_key,
    pinecone_index_name,
    pinecone_cloud,
    pinecone_region,
    embed_model_name,
)

if btn_rebuild or vector_store.is_empty():
    with st.spinner("Xây dựng lại chỉ mục Pinecone…"):
        count = vector_store.rebuild(df)
        st.success(f"Indexed {count} listings → Pinecone index '{pinecone_index_name}'")

# -------------------- Intent extraction --------------------
class Intent:
    def __init__(self, raw: Dict[str, Any]):
        self.max_price_billion = raw.get("max_price_billion")
        self.min_bedrooms = raw.get("min_bedrooms")
        self.min_toilets = raw.get("min_toilets")
        self.must_frontage = raw.get("must_frontage")
        self.keywords = raw.get("keywords", [])

DEFAULT_INTENT_JSON = {
    "max_price_billion": None,
    "min_bedrooms": None,
    "min_toilets": None,
    "must_frontage": False,
    "keywords": []
}

PROMPT_INTENT = (
    "Bạn là bộ trích xuất intent bất động sản. Trả về JSON duy nhất với keys: "
    "max_price_billion (float|null), min_bedrooms (int|null), min_toilets (int|null), "
    "must_frontage (bool), keywords (list[str]). Ví dụ: 'tìm cho tôi nhà dưới 10 tỷ, có 2 phòng ngủ, 2WC và nhà mặt tiền' → "
    "{\"max_price_billion\":10,\"min_bedrooms\":2,\"min_toilets\":2,\"must_frontage\":true,\"keywords\":[]}. Chỉ trả JSON."
)

def parse_intent_with_llm(user_text: str) -> Intent:
    if not (aoai_ep and aoai_key and aoai_dep):
        mx = re.search(r"(dưới|<=?)\s*(\d+[\.,]?\d*)\s*t[ỷy]", user_text, re.I)
        bd = re.search(r"(\d+)\s*(phòng ngủ|pn)", user_text, re.I)
        wc = re.search(r"(\d+)\s*(wc|toilet|vệ sinh)", user_text, re.I)
        frontage = bool(re.search(r"mặt tiền", user_text, re.I))
        return Intent({
            **DEFAULT_INTENT_JSON,
            "max_price_billion": float(mx.group(2).replace(",", ".")) if mx else None,
            "min_bedrooms": int(bd.group(1)) if bd else None,
            "min_toilets": int(wc.group(1)) if wc else None,
            "must_frontage": frontage,
        })
    try:
        from openai import AzureOpenAI
        client_aoai = AzureOpenAI(
            azure_endpoint=aoai_ep,
            api_key=aoai_key,
            api_version="2024-08-01-preview",
        )
        resp = client_aoai.chat.completions.create(
            model=aoai_dep,
            messages=[
                {"role": "system", "content": PROMPT_INTENT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return Intent({**DEFAULT_INTENT_JSON, **data})
    except Exception as e:
        st.warning(f"Azure OpenAI không khả dụng, dùng heuristic: {e}")
        mx = re.search(r"(dưới|<=?)\s*(\d+[\.,]?\d*)\s*t[ỷy]", user_text, re.I)
        bd = re.search(r"(\d+)\s*(phòng ngủ|pn)", user_text, re.I)
        wc = re.search(r"(\d+)\s*(wc|toilet|vệ sinh)", user_text, re.I)
        frontage = bool(re.search(r"mặt tiền", user_text, re.I))
        return Intent({
            **DEFAULT_INTENT_JSON,
            "max_price_billion": float(mx.group(2).replace(",", ".")) if mx else None,
            "min_bedrooms": int(bd.group(1)) if bd else None,
            "min_toilets": int(wc.group(1)) if wc else None,
            "must_frontage": frontage,
        })

# -------------------- Enhanced Vietnamese description --------------------
def vi_enhanced_listing_text(row: pd.Series) -> str:
    loc = str(row.get("Location", "")).strip()
    pr  = str(row.get("Price", "")).strip()
    typ = str(row.get("Type of Hou", row.get("Type of House", ""))).strip()
    beds = row.get("bedrooms", None)
    wc   = row.get("toilets", None)
    floors = row.get("floors", None)
    land = str(row.get("Land Area",""))

    parts = []
    if typ: parts.append(f"Chào mừng bạn đến với {typ.lower()} tại {loc}.")
    else:   parts.append(f"Chào mừng bạn đến với bất động sản tại {loc}.")
    detail = []
    if beds:  detail.append(f"{beds} phòng ngủ")
    if wc:    detail.append(f"{wc} phòng vệ sinh")
    if floors:detail.append(f"{floors} tầng")
    if land:  detail.append(f"diện tích {land}")
    if detail:
        parts.append("Căn nhà có " + ", ".join(detail) + ", phù hợp cho nhu cầu sinh hoạt thoải mái.")
    parts.append("Khu vực thuận tiện kết nối mua sắm, trường học và công viên lân cận, mang lại sự cân bằng giữa tiện nghi đô thị và không gian yên bình.")
    if pr: parts.append(f"Mức giá tham khảo: **{pr}**.")
    parts.append("Đừng bỏ lỡ cơ hội tham khảo bất động sản này!")
    return " ".join(parts)

def llm_vi_enhance_from_rows(rows: List[Dict[str, str]]) -> str:
    if not (aoai_ep and aoai_key and aoai_dep):
        out = []
        for i, r in enumerate(rows, 1):
            out.append(f"**Danh sách {i} - Mô tả nâng cao:**\n" + vi_enhanced_listing_text(pd.Series(r)))
            return "\n\n".join(out)

    try:
        from openai import AzureOpenAI
        client_aoai = AzureOpenAI(azure_endpoint=aoai_ep, api_key=aoai_key, api_version="2024-08-01-preview")
        sys_prompt = (
            "Bạn là trợ lý bất động sản. Hãy viết *markdown* tiếng Việt tự nhiên, thân thiện; "
            "mỗi mục có tiêu đề **Danh sách i - Mô tả nâng cao:** rồi 4–6 câu mô tả. "
            "TUYỆT ĐỐI KHÔNG trả JSON/Array; chỉ prose markdown."
        )
        serialized = json.dumps(rows, ensure_ascii=False)
        resp = client_aoai.chat.completions.create(
            model=aoai_dep,
            messages=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":f"Viết mô tả nâng cao cho các bất động sản sau (JSON): {serialized}"},
            ],
            temperature=0.7,
            max_tokens=900,
        )
        content = resp.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            items = []
            if isinstance(data, list):
                for i, obj in enumerate(data, 1):
                    text = max([str(v) for v in obj.values()], key=len, default="")
                    items.append(f"**Danh sách {i} - Mô tả nâng cao:**\n{text}")
            elif isinstance(data, dict):
                for i, (k, v) in enumerate(data.items(), 1):
                    items.append(f"**Danh sách {i} - Mô tả nâng cao:**\n{v}")
            return "\n\n".join(items) if items else content
        except Exception:
            return content
    except Exception as e:
        st.warning(f"LLM viết mô tả gặp sự cố, dùng mô tả tự động: {e}")
        out = []
        for i, r in enumerate(rows, 1):
            out.append(f"**Danh sách {i} - Mô tả nâng cao:**\n" + vi_enhanced_listing_text(pd.Series(r)))
        return "\n\n".join(out)

# -------------------- Web fallback (Tavily) --------------------
def web_fallback_suggestions(intent: Intent) -> str:
    if not TAVILY_ENABLED or not tavily_client:
        return ("Hiện chưa bật tìm kiếm online (TAVILY_API_KEY). "
                "Vui lòng thêm khóa nếu muốn mình đề xuất nguồn công khai mới nhất.")
    parts = []
    if intent.max_price_billion: parts.append(f"dưới {intent.max_price_billion} tỷ")
    if intent.min_bedrooms:      parts.append(f"{intent.min_bedrooms} phòng ngủ")
    if intent.min_toilets:       parts.append(f"{intent.min_toilets} WC")
    if intent.must_frontage:     parts.append("nhà mặt tiền")
    query = "mua nhà " + ", ".join(parts) + " TP.HCM site:batdongsan.com.vn OR site:chotot.com OR site:alonhadat.com.vn"
    try:
        data = tavily_client.search(query=query, max_results=5, search_depth="basic")
        items = data.get("results", []) if isinstance(data, dict) else []
    except Exception as e:
        return f"Không thể tìm kiếm web lúc này: {e}"
    if not items:
        return "Mình chưa tìm thấy kết quả công khai phù hợp để tham khảo thêm."
    lines = [f"Hiện chưa có kết quả nội bộ phù hợp. Mình đã tham khảo nhanh trên internet ({datetime.now().strftime('%Y-%m-%d')}):"]
    for it in items:
        title = it.get("title", "Nguồn")
        url   = it.get("url", "")
        snip  = (it.get("content", "") or "")[:180].strip()
        lines.append(f"- [{title}]({url}) — {snip}…")
    lines.append("⚠️ Thông tin bên ngoài chỉ tham khảo; vui lòng kiểm tra lại chi tiết & độ cập nhật.")
    return "\n".join(lines)

# -------------------- LangGraph: RAG pipeline --------------------
class ChatState(TypedDict, total=False):
    user_query: str
    intent: Intent
    results_df: pd.DataFrame
    answer: str

def node_intent(state: ChatState) -> ChatState:
    q = state["user_query"]
    intent = parse_intent_with_llm(q)
    return {"intent": intent}

def node_retrieve(state: ChatState) -> ChatState:
    q = state["user_query"]
    intent = state["intent"]
    results = vector_store.search(q.lower(), df, intent, k=30)  # dùng module mới
    return {"results_df": results}

def node_answer(state: ChatState) -> ChatState:
    intent = state["intent"]
    results = state["results_df"]

    if results is None or len(results) == 0:
        answer = web_fallback_suggestions(intent)
        return {"answer": answer}

    top = results.head(2)
    rows_payload = []
    for _, r in top.reset_index(drop=True).iterrows():
        rows_payload.append({
            "Location": str(r.get("Location","")),
            "Price": str(r.get("Price","")),
            "Type of Hou": str(r.get("Type of Hou", r.get("Type of House",""))),
            "bedrooms": r.get("bedrooms"),
            "toilets": r.get("toilets"),
            "floors": r.get("floors"),
            "Land Area": str(r.get("Land Area","")),
        })
    intro = f"Mình tìm thấy **{len(results)}** căn phù hợp. Dưới đây là phần mô tả nâng cao cho {len(rows_payload)} căn nổi bật:"
    body  = llm_vi_enhance_from_rows(rows_payload)
    answer = intro + "\n\n" + body
    return {"answer": answer}

graph = StateGraph(ChatState)
graph.add_node("intent", node_intent)
graph.add_node("retrieve", node_retrieve)
graph.add_node("answer", node_answer)

graph.set_entry_point("intent")
graph.add_edge("intent", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

rag_app = graph.compile()

# -------------------- Chat UI --------------------
st.divider()
st.subheader("💬 Trò chuyện để tìm nhà (Pinecone + LangGraph RAG)")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Nhập nhu cầu. VD: 'tìm cho tôi nhà dưới 10 tỷ, 2 phòng ngủ, 2WC và nhà mặt tiền'")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.spinner("Đang xử lý với RAG (Pinecone + LangGraph)…"):
        final_state = rag_app.invoke({"user_query": user_query})
    answer = final_state.get("answer", "Xin lỗi, mình chưa xử lý được câu hỏi này.")
    st.session_state.history.append({"role": "assistant", "content": answer})

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------- TTS --------------------
st.divider()
st.subheader("🔊 Đọc câu trả lời")
if st.session_state.history:
    last_assistant = next((m for m in reversed(st.session_state.history) if m["role"]=="assistant"), None)
    if last_assistant:
        model, tokenizer, device, sr = load_vits(vits_model_name)
        if st.button("▶️ Phát audio trả lời", use_container_width=True):
            with st.spinner("Đang tổng hợp giọng nói…"):
                wav = tts_generate(model, tokenizer, device, last_assistant["content"])
                audio_bytes = wav_bytes_from_tensor(wav, sr)
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    "⬇️ Tải output.wav",
                    data=audio_bytes,
                    file_name="output.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )

st.caption(
    "Pinecone + LangChain được tách riêng trong vector_store.py. "
    "LangGraph điều phối pipeline intent → retrieval → answer. "
    "Nếu không có kết quả nội bộ, bot fallback sang Tavily (nếu có API key)."
)
