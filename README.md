# RAG知识库问答系统（产品化增强版）

基于 **Streamlit + LangChain + Ollama + ChromaDB** 的本地化 RAG（检索增强生成）知识库问答系统。

**🔒 安全特性**：已部署文件上传防护、输入过滤、异常脱敏、速率限制等安全措施，可安全上传至 GitHub 或部署到公网。

---

## 📋 项目介绍

本项目是一个产品化的 RAG 知识库问答系统，支持多格式文档上传、多文件知识库管理、引用来源展示、知识库持久化等功能。系统完全本地化部署，无需外部 API，保护数据隐私。

**适用场景：**
- 企业知识管理
- 智能客服
- 文档问答
- 个人知识库

---

## ✨ 核心功能

- ✅ **多格式文档支持** — 支持 PDF、TXT、DOCX 三种格式的文件上传和解析
- ✅ **多文件知识库** — 一次上传多个文件，统一检索，支持跨文件问答
- ✅ **引用来源展示** — 回答中标注信息来源（文件名 + 页码），增强可信度
- ✅ **知识库持久化** — 基于 ChromaDB 的持久化存储，关闭重启不丢失
- ✅ **多轮对话记忆** — 支持上下文连续对话，提升交互体验
- ✅ **部署就绪** — 支持本地一键启动，也可部署到 Streamlit Cloud

---

## 🏗️ 技术架构

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit（Web 交互界面） |
| Agent 框架 | LangChain（ReAct Agent + 工具调用） |
| 本地 LLM | Ollama（deepseek-r1:7b） |
| Embedding 模型 | Ollama（shaw/dmeta-embedding-zh） |
| 向量数据库 | ChromaDB（本地持久化） |
| 文档加载 | PyPDFLoader / Docx2txtLoader / TextLoader |

---

## 🚀 快速开始

### 1. 安装 Ollama

下载并安装 Ollama：https://ollama.com

### 2. 下载模型

```bash
ollama pull deepseek-r1:7b
ollama pull shaw/dmeta-embedding-zh
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量（可选）

```bash
cp .env.example .env
# 根据需要编辑 .env 文件
```

### 5. 启动项目

```bash
streamlit run bot_chat_enhanced.py
```

---

## 💡 使用方法

1. 左侧边栏选择「上传新文件构建」模式
2. 上传 PDF/TXT/DOCX 文件（支持多选）
3. 点击「构建知识库」按钮，等待完成
4. 在主界面输入问题，系统会从知识库中检索并回答
5. 回答下方会显示「📚 参考来源」，可查看信息出处

---

## 📁 项目结构

```
rag-knowledge-base/
├── bot_chat_enhanced.py    # 主程序
├── requirements.txt          # Python 依赖
├── .env.example            # 环境变量模板
├── .gitignore             # Git 忽略规则
├── README.md              # 本文件
└── chroma_db/            # （自动生成）知识库持久化目录
```

---

## ⚙️ 环境变量配置

复制 `.env.example` 为 `.env`，并根据需要修改：

```bash
cp .env.example .env
```

主要配置项：
- `OLLAMA_BASE_URL` — Ollama 服务地址（默认 `http://127.0.0.1:11434`）
- `LLM_MODEL` — LLM 模型名称（默认 `deepseek-r1:7b`）
- `EMBED_MODEL` — Embedding 模型名称（默认 `shaw/dmeta-embedding-zh`）
- `CHROMA_PERSIST_DIR` — ChromaDB 持久化目录（默认 `./chroma_db`）
- `RETRIEVAL_TOP_K` — 每次最多检索几个文档片段（默认 `4`）

---

## 🔒 安全特性

本项目已部署以下安全措施：

| 安全类别 | 具体措施 |
|---------|---------|
| **文件安全** | 扩展名白名单、文件大小限制、路径遍历防护 |
| **输入安全** | 长度限制、特殊字符过滤 |
| **输出安全** | 思考标签清理、敏感信息过滤 |
| **资源保护** | Agent 最大迭代次数限制、检索结果数量限制 |
| **异常脱敏** | 错误信息不暴露本地路径和配置 |
| **部署友好** | 全部配置通过环境变量覆盖 |

---

## 📄 许可证

MIT License — 开源免费使用

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) — Agent 框架
- [Streamlit](https://github.com/streamlit/streamlit) — Web 应用框架
- [ChromaDB](https://github.com/chroma-core/chroma) — 向量数据库
- [Ollama](https://ollama.com) — 本地 LLM 运行环境
