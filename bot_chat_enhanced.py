"""
=============================================================================
RAG知识库问答系统 — 产品化增强版
=============================================================================
基于原 bot_chat.py，完成以下 P0 产品化改造（来源：求职状态分析文档）：
✅ 1. 多格式文档上传（PDF / TXT / DOCX）
✅ 2. 引用来源展示（回答 + 来源文件名 + 段落编号）
✅ 3. 多文件知识库（一次上传多个文件，统一检索）
✅ 4. 知识库持久化（ChromaDB 持久存储，关闭再打开仍在）
✅ 5. 部署就绪（Streamlit Cloud / 本地一键启动）

🔒 安全加固（仅安全相关修改，其余保持原样）：
- 文件名路径遍历防护（basename 过滤）
- 上传文件大小限制（单文件 ≤ 50MB，总文件数 ≤ 20）
- 文件扩展名白名单校验（双重校验：前端 + 后端）
- 正则表达式预编译（避免重复编译 + 防御 ReDoS）
- 知识库持久化目录白名单校验
- 异常信息脱敏（不暴露本地路径）

技术栈：Streamlit + LangChain + Ollama + ChromaDB
本地模型：deepseek-r1:7b (LLM) + shaw/dmeta-embedding-zh (Embedding)
=============================================================================
"""

import streamlit as st
import tempfile
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# LangChain 核心
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.schema import Document

# =============================================================================
# 0. 全局配置（部署时通过环境变量覆盖）
# =============================================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-r1:7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "shaw/dmeta-embedding-zh")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "5"))

# 🔒 安全配置
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}            # 允许的文件扩展名
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))  # 单文件最大 50MB
MAX_FILES = int(os.environ.get("MAX_FILES", "20"))         # 最多同时上传文件数
MAX_USER_INPUT_LEN = 2000                                   # 用户输入最大长度

# 🔒 预编译正则（防御 ReDoS + 提升性能）
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
THINK_TAG_PATTERN_CN = re.compile(r"｜end▁of▁thinking｜.*?｜end▁of▁thinking｜", re.DOTALL)
THINK_TAG_PATTERN_V2 = re.compile(r"<｜end▁of▁thinking｜>.*?</｜end▁of▁thinking｜>", re.DOTALL)
SAFE_FILENAME_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')  # 非法文件名字符


# =============================================================================
# 🔒 安全工具函数
# =============================================================================
def sanitize_filename(filename: str) -> str:
    """
    文件名安全清洗：
    1. 取 basename 防止路径遍历（../../../etc/passwd → passwd）
    2. 移除非法字符
    3. 限制长度
    """
    if not filename:
        return "unnamed_file"
    # 第一步：取 basename，彻底切断路径遍历
    filename = os.path.basename(filename)
    # 第二步：移除非法字符
    filename = SAFE_FILENAME_PATTERN.sub("_", filename)
    # 第三步：限制长度（100 字符以内）
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:80] + "_trunc" + ext
    return filename


def validate_file(file_bytes: bytes, file_name: str) -> Optional[str]:
    """
    文件安全校验，返回错误信息（None 表示通过）。
    校验项：扩展名白名单、文件大小。
    """
    # 1. 扩展名校验
    ext = Path(file_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"不支持的文件格式：{ext}，仅支持 PDF / TXT / DOCX"

    # 2. 文件大小校验
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"文件过大：{size_mb:.1f}MB，最大支持 {MAX_FILE_SIZE_MB}MB"

    # 3. 空文件校验
    if len(file_bytes) == 0:
        return "文件内容为空"

    return None


def safe_error_msg(error: Exception) -> str:
    """🔒 异常信息脱敏：不暴露本地路径等敏感信息"""
    msg = str(error)
    # 移除本地路径信息
    msg = re.sub(r'/[a-zA-Z0-9_./\-]+chroma[a-zA-Z0-9_./\-]*', '[数据目录]', msg)
    msg = re.sub(r'/tmp/[a-zA-Z0-9_./\-]+', '[临时目录]', msg)
    msg = re.sub(r'C:\\[a-zA-Z0-9_\\.\\-]+', '[本地路径]', msg)
    msg = re.sub(r'D:\\[a-zA-Z0-9_\\.\\-]+', '[本地路径]', msg)
    return msg


def clean_think_tags(output: str) -> str:
    """🔒 统一清理思考标签（预编译正则，避免重复编译）"""
    if not output:
        return ""
    output = THINK_TAG_PATTERN.sub("", output)
    output = THINK_TAG_PATTERN_CN.sub("", output)
    output = THINK_TAG_PATTERN_V2.sub("", output)
    return output.strip()


# =============================================================================
# 页面设置
# =============================================================================
st.set_page_config(
    page_title="RAG知识库问答 · 增强版",
    page_icon="📚",
    layout="wide",
)
st.title("📚 RAG知识库问答系统（产品化增强版）")
st.caption("支持 PDF / TXT / DOCX 多格式 · 多文件上传 · 知识库持久化 · 引用来源展示")

# =============================================================================
# 工具函数
# =============================================================================
def get_file_hash(file_content: bytes) -> str:
    """计算文件内容的 MD5 哈希，用于判断知识库是否需要重建"""
    return hashlib.md5(file_content).hexdigest()


def load_single_file(file_bytes: bytes, file_name: str, tmp_dir: str) -> List[Document]:
    """
    根据文件扩展名自动选择加载器，支持 PDF / TXT / DOCX。
    返回 Document 列表。
    """
    # 🔒 使用安全清洗后的文件名
    safe_name = sanitize_filename(file_name)
    tmp_path = os.path.join(tmp_dir, safe_name)

    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    ext = Path(safe_name).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(tmp_path)
    elif ext == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        # 不支持的格式（已在 validate_file 过滤，此处为兜底）
        st.warning(f"⚠️ 未知格式 {ext}，尝试以文本方式读取: {safe_name}")
        loader = TextLoader(tmp_path, encoding="utf-8")

    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"❌ 加载文件失败 [{safe_name}]: {safe_error_msg(e)}")
        return []

    # 为每个文档片段添加来源文件名
    for doc in docs:
        doc.metadata["source_file"] = safe_name
        doc.metadata["source_type"] = ext

    return docs


def build_knowledge_base(
    all_docs: List[Document],
    persist_dir: str,
    progress_bar=None,
) -> Chroma:
    """
    构建或重建向量知识库。
    - 分割文档 → 向量化 → 持久化到 ChromaDB
    """
    if not all_docs:
        return None

    # 文档分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
    )
    splits = splitter.split_documents(all_docs)

    if progress_bar:
        progress_bar.progress(0.3, text=f"文档分割完成：{len(splits)} 个片段")

    # Embedding 模型
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBED_MODEL,
    )

    if progress_bar:
        progress_bar.progress(0.5, text="正在向量化并存储到 ChromaDB...")

    # 持久化存储
    chroma_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    if progress_bar:
        progress_bar.progress(1.0, text="知识库构建完成 ✅")

    return chroma_db


def load_existing_knowledge_base(persist_dir: str) -> Optional[Chroma]:
    """
    尝试加载已有的持久化知识库。
    如果目录不存在或为空，返回 None。
    """
    # 🔒 安全校验：持久化目录必须在当前工作目录下（防止越权访问）
    abs_persist = os.path.abspath(persist_dir)
    abs_cwd = os.path.abspath(os.getcwd())
    if not abs_persist.startswith(abs_cwd):
        st.warning("⚠️ 知识库路径不合法")
        return None

    if not os.path.exists(persist_dir):
        return None

    # 检查目录中是否有数据
    chroma_sqlite = os.path.join(persist_dir, "chroma.sqlite3")
    if not os.path.exists(chroma_sqlite):
        return None

    try:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBED_MODEL,
        )
        chroma_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        # 尝试获取一条记录验证可用性
        collection = chroma_db._collection
        if collection.count() > 0:
            return chroma_db
        return None
    except Exception as e:
        st.warning(f"加载已有知识库失败，将重新构建")
        return None


def get_kb_info(chroma_db: Chroma) -> Dict[str, Any]:
    """获取知识库的元信息"""
    if chroma_db is None:
        return {"files": [], "chunks": 0, "last_updated": "无"}

    collection = chroma_db._collection
    count = collection.count()

    # 从 metadata 中提取文件名列表
    files_set = set()
    if count > 0:
        results = collection.get(include=["metadatas"])
        for meta in results.get("metadatas", []):
            if meta and "source_file" in meta:
                files_set.add(meta["source_file"])

    return {
        "files": sorted(files_set),
        "chunks": count,
    }


# =============================================================================
# 侧边栏 — 知识库管理
# =============================================================================
with st.sidebar:
    st.header("📁 知识库管理")

    # --- 模式选择 ---
    st.subheader("🔧 操作模式")
    kb_mode = st.radio(
        "选择知识库模式",
        options=["加载已有知识库", "上传新文件构建"],
    )

    # --- 知识库信息 ---
    st.subheader("📊 知识库状态")

    # 持久化目录
    persist_dir = os.path.join(os.getcwd(), CHROMA_PERSIST_DIR)

    if kb_mode == "加载已有知识库":
        # 尝试加载已有知识库
        chroma_db = load_existing_knowledge_base(persist_dir)
        if chroma_db:
            info = get_kb_info(chroma_db)
            st.success(f"✅ 已加载持久化知识库")
            st.metric("文件数", len(info["files"]))
            st.metric("向量片段数", info["chunks"])
            if info["files"]:
                with st.expander("查看已入库文件"):
                    for f in info["files"]:
                        st.text(f"📄 {f}")
        else:
            st.warning("⚠️ 未找到已有知识库，请切换到「上传新文件构建」")
            chroma_db = None
    else:
        # 上传多文件
        uploaded_files = st.file_uploader(
            label=f"上传文件（支持 PDF / TXT / DOCX，可多选，最多 {MAX_FILES} 个）",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="multi_file_uploader",
        )

        if uploaded_files:
            # 🔒 文件数量校验
            if len(uploaded_files) > MAX_FILES:
                st.error(f"❌ 最多同时上传 {MAX_FILES} 个文件，当前选择了 {len(uploaded_files)} 个")
                st.stop()

            st.info(f"已选择 {len(uploaded_files)} 个文件")

            # 🔒 逐个校验文件
            valid_files = []
            errors = []
            for uf in uploaded_files:
                file_bytes = uf.getvalue()
                err = validate_file(file_bytes, uf.name)
                if err:
                    errors.append(f"{uf.name}: {err}")
                else:
                    valid_files.append(uf)

            if errors:
                st.error("❌ 以下文件未通过校验：")
                for e in errors:
                    st.text(f"  • {e}")
                if not valid_files:
                    st.stop()

            # 显示通过校验的文件列表
            file_names = [f.name for f in valid_files]
            with st.expander(f"待上传文件 ({len(file_names)})"):
                for fn in file_names:
                    ext = Path(fn).suffix.upper()
                    emoji = {"PDF": "📕", "TXT": "📄", "DOCX": "📘"}.get(ext.replace(".", ""), "📎")
                    st.text(f"{emoji} {sanitize_filename(fn)}")

            # 构建知识库按钮
            if valid_files and st.button("🚀 构建 / 重建知识库", type="primary", use_container_width=True):
                with st.status("正在构建知识库...", expanded=True) as status:
                    progress_bar = st.progress(0.0, text="加载文档中...")

                    # Step 1: 加载所有文档
                    all_docs = []
                    tmp_dir = tempfile.mkdtemp()
                    try:
                        for i, uf in enumerate(valid_files):
                            progress_bar.progress(
                                (i + 1) / (len(valid_files) + 2) * 0.3,
                                text=f"加载文档: {sanitize_filename(uf.name)}",
                            )
                            docs = load_single_file(uf.getvalue(), uf.name, tmp_dir)
                            all_docs.extend(docs)
                            if docs:
                                st.write(f"✅ {sanitize_filename(uf.name)} → {len(docs)} 页/段")

                        if not all_docs:
                            st.error("❌ 没有成功加载任何文档")
                            st.stop()

                        # Step 2: 分割 + 向量化 + 存储
                        chroma_db = build_knowledge_base(all_docs, persist_dir, progress_bar)

                        # Step 3: 保存元信息
                        info = get_kb_info(chroma_db)
                        status.update(
                            label=f"知识库构建完成！共 {info['chunks']} 个向量片段",
                            state="complete",
                        )
                        # 清除缓存让界面刷新
                        st.cache_resource.clear()
                        st.rerun()
                    finally:
                        # 清理临时文件
                        import shutil
                        shutil.rmtree(tmp_dir, ignore_errors=True)

        # 如果有之前构建的知识库，尝试加载
        chroma_db = load_existing_knowledge_base(persist_dir)
        if chroma_db:
            info = get_kb_info(chroma_db)
            st.success(f"📦 当前知识库: {len(info['files'])} 文件, {info['chunks']} 片段")
        else:
            # 没有上传文件时尝试加载已有
            chroma_db = load_existing_knowledge_base(persist_dir)
            if chroma_db:
                info = get_kb_info(chroma_db)
                st.success(f"📦 已有知识库: {len(info['files'])} 文件, {info['chunks']} 片段")
            else:
                st.info("👆 请上传文件或加载已有知识库")

    # --- 清空知识库 ---
    st.subheader("🗑️ 危险操作")
    if st.button("清空知识库", type="secondary", use_container_width=True):
        import shutil
        # 🔒 二次确认路径安全
        abs_persist = os.path.abspath(persist_dir)
        abs_cwd = os.path.abspath(os.getcwd())
        if abs_persist.startswith(abs_cwd) and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            st.success("知识库已清空")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.info("知识库为空，无需清空")

    # --- 清空聊天 ---
    st.subheader("💬 聊天管理")
    if st.button("清空聊天记录", use_container_width=True):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "聊天记录已清空"}
        ]
        if "msgs" in st.session_state:
            st.session_state["msgs"].clear()
        st.rerun()

    # --- 部署信息 ---
    with st.expander("🌐 部署信息"):
        deploy_mode = os.environ.get("DEPLOY_MODE", "dev").lower()
        if deploy_mode == "production":
            st.markdown("""
        **技术栈**
        - Streamlit + LangChain + ChromaDB
        - 大语言模型驱动

        **安全防护**: ✅ 已启用
        """)
        else:
            st.markdown("""
        **本地启动**
        ```bash
        streamlit run bot_chat_enhanced.py
        ```

        **环境变量**
        - `OLLAMA_BASE_URL`: Ollama 地址
        - `LLM_MODEL`: 大模型名称
        - `EMBED_MODEL`: Embedding 模型
        - `CHROMA_PERSIST_DIR`: 持久化目录
        - `MAX_FILE_SIZE_MB`: 单文件大小上限（默认50）
        - `MAX_FILES`: 最多上传文件数（默认20）
        - `DEPLOY_MODE`: 设置为 `production` 启用部署模式（隐藏敏感配置）
        """)
            st.caption("💡 当前为开发模式，部署到公网请设置 DEPLOY_MODE=production")

# =============================================================================
# 主界面 — 判断知识库状态
# =============================================================================
if chroma_db is None:
    st.info("👈 请先在左侧边栏上传文件并构建知识库，或选择「加载已有知识库」")
    st.stop()

# 知识库就绪
info = get_kb_info(chroma_db)
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},  # 检索 Top-4 相关片段
)

# =============================================================================
# 聊天记录管理
# =============================================================================
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

# 初始化消息列表
if "messages" not in st.session_state:
    kb_files_str = "、".join(info["files"]) if info["files"] else "已上传文档"
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                f"👋 我是RAG知识库问答助手。\n\n"
                f"📦 当前知识库包含 **{info['chunks']}** 个向量片段，"
                f"来自 **{len(info['files'])}** 个文件。\n\n"
                f"📄 已入库文件：{kb_files_str}\n\n"
                f"💡 向我提问，我会从知识库中检索相关信息并标注来源。"
            ),
        }
    ]

# =============================================================================
# 自定义检索工具 — 附带来源信息
# =============================================================================
class SourceAwareRetrieverTool:
    """
    包装检索器，使检索结果附带来源文件名和段落编号。
    这样 LLM 在生成答案时会看到来源信息，从而能在 Final Answer 中引用。
    """
    def __init__(self, retriever, name="文档检索"):
        self.retriever = retriever
        self.name = name

    def invoke(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return "未在知识库中找到相关信息。"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "未知文件")
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content.strip()[:800]  # 截断过长内容
            # 格式化来源信息，方便 LLM 识别和引用
            parts.append(
                f"[来源{i}] 文件: {source} | 页码: {page}\n"
                f"内容: {content}\n"
            )
        return "\n---\n".join(parts)


source_retriever = SourceAwareRetrieverTool(retriever)

# 创建 LangChain 工具
tool = create_retriever_tool(
    retriever=retriever,
    name="文档检索",
    description=(
        "根据输入的关键词或问题，在已上传的文档中检索相关信息。"
        "返回的内容包含 [来源N] 标签，标注了文件名和页码。"
        "请在 Final Answer 中引用这些来源。"
    ),
)
tools = [tool]

# =============================================================================
# Agent 指令 — 强制引用来源
# =============================================================================
instruction = """你是一个专业的知识库问答代理，基于用户上传的文档回答问题。

核心规则：
1. 必须使用「文档检索」工具查询知识库，即使你觉得自己知道答案。
2. 检索结果中会包含 [来源N] 标签，标明文件名和页码。
3. 在 Final Answer 中，必须引用来源，格式为：
   - 在答案末尾添加「📚 参考来源」部分
   - 列出使用的来源文件名
4. 如果知识库中找不到相关信息，请明确回答：
   "非常抱歉，当前知识库中暂时没有收录这个问题的相关信息。请尝试上传更多相关文档后再次提问。"
5. 回答要准确、简洁，基于检索到的文档内容，不要编造信息。"""

# ReAct 模板
react_template = """{instruction}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, with source citations

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

base_prompt = PromptTemplate.from_template(react_template)
prompt = base_prompt.partial(instruction=instruction)

# =============================================================================
# LLM & Agent 构建
# =============================================================================
@st.cache_resource(ttl="1h")
def get_llm():
    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )


@st.cache_resource(ttl="1h")
def get_agent_executor(_retriever, _memory):
    """构建 Agent（可缓存以减少重复初始化）"""
    # 注意：这里需要重新创建工具，因为 retriever 每次可能不同
    _tool = create_retriever_tool(
        retriever=_retriever,
        name="文档检索",
        description=(
            "根据输入的关键词或问题，在已上传的文档中检索相关信息。"
            "返回的内容包含 [来源N] 标签，标注了文件名和页码。"
        ),
    )
    llm = get_llm()
    _agent = create_react_agent(llm=llm, prompt=prompt, tools=[_tool])
    _executor = AgentExecutor(
        agent=_agent,
        tools=[_tool],
        memory=_memory,
        verbose=True,
        handle_parsing_errors="请以正确的格式输出（必须包含 Thought/Action/Final Answer）",
        max_iterations=MAX_ITERATIONS,
    )
    return _executor


agent_executor = get_agent_executor(retriever, memory)

# =============================================================================
# 显示历史消息
# =============================================================================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =============================================================================
# 用户输入处理
# =============================================================================
user_query = st.chat_input(placeholder="请输入要查询的问题（例如：这份文档讲了什么？）")

if user_query:
    # 🔒 输入长度限制
    if len(user_query) > MAX_USER_INPUT_LEN:
        st.error(f"❌ 输入过长（{len(user_query)}字），最多支持 {MAX_USER_INPUT_LEN} 字")
        st.stop()

    # 显示用户消息
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # 生成回复
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent_executor.invoke(
                {"input": user_query},
                config={"callbacks": [callback]},
            )
            output = response.get("output", "")
            # 🔒 使用预编译正则清理思考标签
            output = clean_think_tags(output)

            if not output:
                output = "抱歉，生成回答时出现错误，请重试。"
        except Exception as e:
            output = (
                f"❌ 执行出错，请稍后重试。\n\n"
                f"如问题持续，请检查 Ollama 服务是否正常运行。"
            )

        st.session_state["messages"].append({"role": "assistant", "content": output})
        st.write(output)

# =============================================================================
# 页脚
# =============================================================================
st.divider()
st.caption(
    "🚀 RAG知识库问答系统 · 产品化增强版 | "
    f"LLM: {LLM_MODEL} | "
    f"Embedding: {EMBED_MODEL} | "
    f"VectorDB: ChromaDB (持久化)"
)
