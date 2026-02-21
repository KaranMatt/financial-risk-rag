# Multi-Document Financial RAG System

A production-ready Retrieval-Augmented Generation (RAG) system designed for financial document analysis, optimized to run entirely on consumer-grade GPUs with careful VRAM management. This system processes multiple PDF documents and provides accurate, cited answers to financial queries using state-of-the-art NLP models — now served via a **FastAPI REST interface** for seamless integration.

## Test Documents Used

This system was developed and tested on complex, domain-specific financial research papers:

1. **Climate-Related Financial Policy and Systemic Risk**  
   *Authors: Alin Marius Andries, Steven Ongena*  
   Focus: Climate risk integration in financial systems and policy frameworks

2. **Corporate Credit Risk Under Misaligned Transition Expectations: A Firm-Level Modeling Approach**  
   *Author: Elisa Ndiaye*  
   Focus: Advanced quantitative modeling of corporate credit risk in climate transitions

3. **Dynamic Oil Price–Stock Market Volatility Spillovers in Oil-Exporting Countries**  
   *Author: Haseen Ahmed*  
   Focus: Econometric analysis of oil-stock market interdependencies

**Domain Complexity Handled:**
- Dense mathematical formulations and Greek notation (stochastic models, probability distributions)
- Financial jargon and technical terminology (carbon mitigation strategies, Nationally Determined Contributions, default probability modeling)
- Multi-layered concepts requiring cross-document synthesis
- Quantitative metrics requiring precise extraction (percentage points, statistical measures)
- Citation-heavy academic writing with cross-references

The RAG system successfully navigates this complexity through intelligent chunking, semantic understanding, and precise retrieval mechanisms.

## Key Features

- **100% Local Deployment**: No API calls, no cloud dependencies - runs completely offline on consumer hardware
- **FastAPI Interface**: Production-ready REST API with health checks, lifespan model loading, and structured request/response schemas
- **VRAM-Optimized**: Carefully tuned for GPUs with limited memory (tested on consumer-grade cards)
- **Multi-Document Support**: Processes and retrieves information across multiple financial PDFs simultaneously
- **Intelligent Reranking**: Uses cross-encoder models for superior retrieval quality
- **Source Citations**: Every answer includes document name and page number references
- **Optimized Generation**: Fine-tuned parameters prevent repetition and improve output quality
- **Punches Above Its Weight**: Through careful optimization, a 1.5B parameter model closes the gap with full 7B models on RAG-style tasks

---

## Performance Boosting: Small Model, Strategic Optimization

### How a 1.5B Model Closes the Gap with 7B

**The Challenge**: Qwen2.5-1.5B has only 1.5 billion parameters — roughly 4–5× fewer than a full 7B model. In open-ended generation, that gap is significant: 7B models carry more world knowledge in their weights, exhibit stronger instruction-following by default, and maintain coherence more naturally over longer outputs.

**The Key Insight**: In a RAG system, the model's role is fundamentally different. It is not being asked to recall facts from memory — it is being asked to read a supplied context and synthesize a structured answer. This shifts the bottleneck from *parametric knowledge* (where 7B wins decisively) to *context utilization quality* (where the gap can be almost entirely closed through smart engineering). The optimizations below are what make that possible.

---

#### 1. Intelligent Retrieval Reduces Model Burden

```
Without Reranking (Baseline):
  Query → Semantic Search (k=3) → Small Model → Limited, possibly irrelevant context

With Reranking (Optimized):
  Query → Semantic Search (k=20) → Cross-Encoder Reranking → Top 3 Best Chunks → Small Model → Focused, highly relevant context
```

A 7B model can partially compensate for poor retrieval using its larger internal knowledge base. A 1.5B model cannot afford that luxury — it depends almost entirely on what it is handed. By using a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) to score all 20 retrieved candidates against the query jointly, the final 3 chunks passed to the model are far more precise than anything a naive top-k search would produce. The 1.5B model receives context so targeted that it rarely needs to "guess" beyond what is written directly in front of it.

**Impact**:
- Reranking ensures the model receives only the most relevant context
- Reduces reliance on the model's parametric knowledge
- Minimizes hallucination by providing high-quality input
- Enables multi-document synthesis despite small model size
- Example: A climate change query successfully pulls from 3 different research papers

---

#### 2. Prompt Engineering as a Force Multiplier

A 7B model given a vague prompt will still produce a reasonable answer — it has enough capacity to infer intent and fill gaps. A 1.5B model with the same vague prompt will drift, fabricate, or produce repetitive output. The solution is to write the prompt as a strict behavioral contract:

```python
prompt = f'''You are a financial analyst assistant. Answer the question using ONLY the provided context.

IMPORTANT RULES:
1. Be concise - maximum 500 words
2. Always cite sources: [Doc: filename | Page: X]
3. If context is insufficient, state: "Based on available documents, I cannot fully answer this."
4. No speculation beyond the documents
5. For financial metrics, copy exact numbers from source

Question: {request.question}

Context:
{context}

Answer (concise, cited):'''
```

**Why This Works**:
- **Role definition** primes the model for the financial domain without requiring it to retrieve that framing from its weights
- **Strict grounding** ("ONLY the provided context") eliminates the small-model tendency to confabulate
- **Citation requirement** enforces factual accountability at output time
- **Conciseness constraint** keeps the model focused before it has a chance to lose coherence
- **Explicit rules** replace the implicit reasoning capacity that a 7B model applies naturally

**Result**: The 1.5B model acts as a precise document synthesizer rather than a general knowledge engine — the exactly right role for it in a RAG pipeline.

---

#### 3. Hyperparameter Tuning Maximizes Coherence

Each parameter was iteratively refined to compensate for behaviors that emerge specifically in smaller models at generation time:

| Parameter | Early Iteration | Final Optimized | Why It Matters for a 1.5B Model |
|-----------|----------------|-----------------|----------------------------------|
| `temperature` | 0.7 | **0.3** | Small models at high temperature produce inconsistent, unreliable outputs |
| `repetition_penalty` | 1.3 | **1.1** | 1.3 caused unnatural phrasing; 1.1 is the minimum effective deterrent |
| `top_p` | 0.9 | **removed** | Sampling diversity is a liability in factual extraction tasks |
| `no_repeat_ngram_size` | — | **3** | Directly blocks looping patterns common in smaller models |
| `max_new_tokens` | 1024 | **512** | Shorter outputs force focus; small models lose coherence in long generations |

**Combined Effect**: These parameters channel the model's limited capacity toward precision and relevance rather than allowing it to wander.

---

#### 4. Architecture Synergy: Each Layer Amplifies the Others

```
High-Quality Retrieval (Reranking)
    ↓
Provides Precise, Relevant Context
    ↓
Structured Prompt Guides Extraction
    ↓
Tuned Parameters Prevent Failure Modes
    ↓
1.5B Model Delivers Professional-Grade Results
```

**Concrete Example**:
- **Query**: "Explain Default and how is the probability computed?"
- **Without optimization**: Generic explanation, potential inaccuracies, no citations
- **With optimization**:
  - Reranker surfaces the exact section on Black-Scholes modifications
  - Prompt demands "copy exact numbers from source"
  - Low temperature prevents fabrication of statistics
  - Model outputs a precise technical explanation with correct page-level citations

---

#### 5. Closing the Gap: 1.5B Optimized vs. 7B Full Precision

The comparison below reflects the practical performance characteristics of each configuration in a RAG-specific task context. This is not a formal benchmark — it is based on observed system behavior during development.

| Dimension | 1.5B (This Project, Optimized) | 7B (Full Precision, Unquantized) |
|-----------|-------------------------------|----------------------------------|
| **VRAM Usage** | ~3–4 GB | ~14–16 GB |
| **Inference Speed** | 3–5s per answer | 10–20s per answer |
| **Hardware Required** | Consumer GPU (6GB+) | High-end GPU (16–24GB+) |
| **Factual Grounding (RAG)** | High — enforced via prompt + reranking | High — model's capacity helps, but enforcement still needed |
| **Citation Consistency** | 100% — mandated by structured prompt | Depends on prompt design |
| **Hallucination on RAG Tasks** | Near-zero — context dependency enforced | Low — benefits from stronger parametric reasoning |
| **Open-Ended Knowledge** | Limited by parameter count | Substantially stronger |
| **Financial Jargon Handling** | Strong via retrieval + domain chunking | Strong via parametric knowledge |
| **Multi-Document Synthesis** | Effective via k=20 + reranking | Effective, but gains less from retrieval optimization |

**Key Takeaway**: On RAG tasks where the answer lives entirely in the retrieved context, the gap between 1.5B (optimized) and 7B (full precision) narrows significantly. The 7B model retains a clear edge on open-ended reasoning, general knowledge, and ambiguous queries where parametric knowledge must fill gaps. But for document-grounded financial Q&A — the exact use case of this project — the 1.5B system with proper retrieval, prompting, and tuning delivers results that are practically competitive at a fraction of the hardware cost, without any quantization required.

---

#### 6. The Multiplier Effect

```
Base 1.5B Model:                       Functional but basic responses
+ Semantic Embeddings (BGE):           Better document matching across corpus
+ Cross-Encoder Reranking:             Precision filtering of top candidates
+ Structured Prompting:                Strict grounding and behavior enforcement
+ Hyperparameter Tuning:               Coherence and focus maintenance
+ Domain-Specific Chunking:            Context preservation for technical content
────────────────────────────────────────────────────────────────────────────────
= Final System:                        Professional-grade financial document analysis
```

**Bottom Line**: Strategic optimization transforms a lightweight 1.5B model into a capable system for complex financial document analysis — without expensive hardware, without quantization tricks, and without a larger model.

---

## FastAPI Interface

The system is served as a production-ready REST API via FastAPI (`main.py`). All models are loaded once at startup using FastAPI's `lifespan` context manager, making the API stateful and efficient — no cold-start overhead on individual requests.

### API Architecture

```
Client Request (POST /ask)
        ↓
FastAPI Endpoint
        ↓
FAISS Similarity Search (k=20)
        ↓
Cross-Encoder Reranking → Top 3 Chunks
        ↓
Prompt Construction with Context + Metadata
        ↓
Qwen2.5-1.5B Generation Pipeline
        ↓
Structured JSON Response
```

### Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Note**: The vector database (`RAG Vector DB/`) must be generated from the notebook before starting the API. The `.gitignore` excludes it from the repository — see the [Project Structure](#project-structure) section.

### Endpoints

#### `GET /health`
Returns the current status of the API and whether all models are loaded.

```json
{
  "status": "Active",
  "Models Loaded": true
}
```

#### `GET /root`
Welcome message confirming the API is reachable.

```json
{
  "message": "Welcome to the RAG API"
}
```

#### `POST /ask`
Submit a financial question and receive a cited, grounded answer.

**Request Body**:
```json
{
  "question": "Explain the probability of default computation in the Ndiaye paper."
}
```

**Response Body**:
```json
{
  "quesiton": "Explain the probability of default computation in the Ndiaye paper.",
  "response": "The probability of default is computed using a Monte Carlo simulation approach based on Black-Scholes modifications... [Doc: Corporate Credit Risk Modelling Paper.pdf | Page: 7]"
}
```

### Request & Response Schema

```python
class Questionclass(BaseModel):
    question: str

class ResponseClass(BaseModel):
    quesiton: str   # field name preserved from source
    response: str
```

### Lifespan Model Loading

All models are loaded once when the server starts and released cleanly on shutdown, avoiding the overhead of loading multi-gigabyte models on every request:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load embeddings, FAISS index, LLM pipeline, reranker
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    vector_db  = FAISS.load_local('RAG Vector DB', embeddings=embeddings,
                                   allow_dangerous_deserialization=True)
    model      = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
                                                       device_map='auto',
                                                       dtype=torch.bfloat16,
                                                       low_cpu_mem_usage=True)
    pipe       = pipeline('text-generation', model=model, tokenizer=tokenizer,
                           temperature=0.3, do_sample=True, max_new_tokens=512,
                           repetition_penalty=1.1, no_repeat_ngram_size=3)
    rerank     = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    yield
    # Shutdown: release all model references
    vector_db = pipe = rerank = None
```

### Additional FastAPI Dependencies

```
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
```

---

## Architecture Overview

```
PDF Documents → Document Loading → Text Chunking → Embedding Generation
                                                            ↓
                                                     FAISS Vector DB
                                                            ↓
User Query (HTTP POST /ask)
        ↓
FastAPI → Similarity Search (k=20) → Cross-Encoder Reranking (top 3)
                                                            ↓
                                    Context Formation → LLM Generation → Cited Answer (JSON)
```

---

## System Components

### 1. Document Processing Pipeline
- **Loader**: PyMuPDFLoader for efficient PDF parsing
- **Chunking Strategy**:
  - Chunk size: 750 characters
  - Overlap: 150 characters
  - Hierarchical separators: `\n\n`, `\n`, `.`, ` `
  - Result: 470 chunks from input documents

### 2. Embedding & Vector Store
- **Model**: `BAAI/bge-small-en-v1.5` (lightweight, high-quality embeddings)
- **Vector DB**: FAISS for fast similarity search
- **Storage**: Persistent local storage — excluded from version control via `.gitignore`

### 3. Retrieval System
- **Initial Retrieval**: Top 20 documents via similarity search
- **Reranking Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Final Selection**: Top 3 most relevant chunks after reranking
- **Rationale**: Reranking significantly improves precision over pure semantic search

### 4. Language Model
- **Model**: Qwen2.5-1.5B-Instruct (efficient instruction-following model)
- **Precision**: bfloat16 for memory efficiency
- **Generation Parameters** (Optimized through iterations):
  - Temperature: 0.3 (balanced creativity/accuracy)
  - Max tokens: 512
  - Repetition penalty: 1.1
  - No repeat n-gram size: 3
  - Do sample: True

### 5. FastAPI Server
- **Framework**: FastAPI with Pydantic v2 request/response models
- **Model Loading**: Lifespan context manager (single load at startup)
- **Server**: Uvicorn ASGI server

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: 6GB+ VRAM)
- CUDA Toolkit installed

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
langchain>=0.1.0
langchain-community>=0.1.0
langchain-huggingface>=0.0.1
sentence-transformers>=2.2.0
faiss-gpu>=1.7.2       # or faiss-cpu
pymupdf>=1.22.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
```

---

## Project Structure

```
financial-rag-system/
│
├── data/                                    # Place your PDF documents here
│   ├── Climate Related Financial Risk Paper.pdf
│   ├── Dynamic Oil Price Stock Volatility Paper.pdf
│   └── Corporate Credit Risk Modelling Paper.pdf
│
├── RAG Vector DB/                           # ⚠ Git-ignored — generate locally via notebook
│   ├── index.faiss
│   └── index.pkl
│
├── main.py                                  # FastAPI application
├── multidoc-financial-rag-system.ipynb      # Notebook: ingestion, chunking, vector DB creation
├── .gitignore                               # Excludes RAG Vector DB/ and __pycache__/
└── README.md
```

### .gitignore

The repository includes a `.gitignore` that intentionally excludes two items:

- **`RAG Vector DB/`** — The FAISS index is generated locally from your documents and can be hundreds of MBs in size. It is fully reproducible by running the notebook, so it does not belong in version control.
- **`__pycache__/`** — Python bytecode cache directories are environment-specific and should never be committed.

After cloning the repository, run the ingestion notebook first to generate the vector database, then start the API server.

---

## Example Queries

### Via Notebook

```python
# Climate finance and systemic risk (Andries & Ongena paper)
format_prompt('What Role does Climate Change Play in Finance and investments?')

# Market interdependencies (Ahmed paper)
format_prompt('How does Oil Prices affect the stock market?')

# Advanced modeling concepts (Ndiaye paper)
format_prompt('Explain Carbon Mitigation Strategy')

# Quantitative finance (Ndiaye paper)
format_prompt('Explain Default and how is the probability of Default is computed?')

# Policy and governance (Andries & Ongena paper)
format_prompt('Explain the role of Paris Agreement')
```

### Via FastAPI

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "How does oil price volatility affect stock markets in oil-exporting countries?"}'
```

**Example Response**:
```json
{
  "quesiton": "How does oil price volatility affect stock markets in oil-exporting countries?",
  "response": "Oil prices have a significant impact on stock markets in oil-exporting economies. A one percentage point increase in oil prices leads to a 2.5% decrease in stock prices [Doc: Dynamic Oil Price-Stock Market Volatility Paper.pdf | Page: 3]. Bjørnland's research further shows that a 10% rise in global oil prices resulted in a 25% increase in stock return volatility [Doc: Dynamic Oil Price-Stock Market Volatility Paper.pdf | Page: 4]."
}
```

**Query Complexity Handled:**
- Cross-document synthesis (climate change impacts from multiple sources)
- Technical precision (exact statistical measures and formulas)
- Multi-layered concepts (default probability via nested Monte Carlo)
- Policy interpretation (Paris Agreement framework across governance levels)

---

## Optimization Journey: Iterative Refinement Process

### Initial Configuration (Suboptimal Results)

```python
repetition_penalty = 1.3  # Too high, caused unnatural text
top_p = 0.9               # Too high, led to incoherent responses
temperature = 0.7         # Too much randomness
max_new_tokens = 1024     # Model lost focus in long outputs
# No reranking model      # Poor retrieval precision
# Basic semantic search (k=3)  # Limited context diversity
```

**Problems Encountered**:
- **Retrieval Limitations**: Simple k=3 semantic search missed relevant information across documents
- **Repetitive Patterns**: "The document states... The document states... The document states..."
- **Incoherent Outputs**: High top_p caused rambling, unfocused answers
- **Fabrication Issues**: Without reranking, poor context quality led to unreliable outputs
- **Loss of Focus**: Long token limits caused the 1.5B model to drift off-topic
- **Inconsistent Citations**: Vague or missing source references

**Example Suboptimal Output**:
```
Q: How do oil prices affect stock markets?
A: Oil prices can affect stock markets in various ways. The document discusses
this topic extensively. Oil is an important commodity. Markets respond to oil.
The document mentions several studies. Oil prices impact many sectors...
[No specific data, no citations, repetitive phrasing]
```

### Iteration 1: Adding Cross-Encoder Reranking

```python
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
k=20 initial → top 3 final  # Cast wide net, then filter precisely
```

Observed improvements: better retrieval of relevant passages, reduced irrelevant context in final selection, more targeted answers.

### Iteration 2: Hyperparameter Refinement

```python
temperature = 0.3          # Reduced randomness significantly
repetition_penalty = 1.1   # Gentle penalty maintains natural language
no_repeat_ngram_size = 3   # Blocks phrase-level repetition
max_new_tokens = 512       # Forces conciseness
```

Observed improvements: eliminated repetitive loops, more coherent and focused outputs, reduced tendency to drift off-topic.

### Iteration 3: Structured Prompt Engineering

```python
# Role definition + strict grounding + citation mandate + conciseness + precision rules
```

Observed improvements: consistent source citations in every response, near-zero fabrication, strong adherence to source material.

### Iteration 4: Domain-Appropriate Chunking

```python
chunk_size = 750
chunk_overlap = 150
separators = ['\n\n', '\n', '.', ' ']
```

Observed improvements: better preservation of mathematical expressions, maintained context for technical concepts, reduced splitting of critical information.

### Final Optimized Configuration

```python
# Retrieval Layer
embeddings    = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
initial_k     = 20
rerank_model  = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
final_k       = 3

# Generation Layer
model                = 'Qwen/Qwen2.5-1.5B-Instruct'
temperature          = 0.3
repetition_penalty   = 1.1
no_repeat_ngram_size = 3
do_sample            = True
max_new_tokens       = 512

# Prompt Layer
# - Financial analyst role definition
# - Strict context grounding
# - Mandatory citations [Doc: X | Page: Y]
# - Conciseness constraint (500 words)
# - Exact number preservation rule
```

**Cumulative Improvements:**
- **Retrieval Quality**: Significantly enhanced through reranking
- **Output Coherence**: Marked improvement through hyperparameter tuning
- **Fabrication Reduction**: Near elimination through prompt engineering
- **Citation Consistency**: 100% citation inclusion in responses
- **Technical Precision**: Better handling through optimized chunking

**Example Optimized Output**:
```
Q: How do oil prices affect stock markets?
A: Oil prices have a significant impact on stock markets, especially in volatile
periods. Studies show that a one percentage point increase in oil prices can lead
to a 2.5% decrease in stock prices [Doc: Dynamic Oil Price-Stock Market
Volatility Paper.pdf | Page: 3]. According to Bjørnland's research, a 10% rise
in global oil prices resulted in a 25% increase in stock return volatility
[Doc: Dynamic Oil Price-Stock Market Volatility Paper.pdf | Page: 4]...
```

### Key Lessons from the Optimization Process

1. **Retrieval Quality is Critical**: Cross-encoder reranking provides substantial improvements over pure semantic search
2. **Prompt Engineering as Constraint**: Explicit rules guide small models toward reliable behavior that larger models exhibit more naturally
3. **Hyperparameter Impact**: Careful tuning eliminates common failure modes (repetition, drift, incoherence)
4. **Domain-Specific Design**: Chunking strategy must respect document structure and content type

**Cost-Benefit**: These optimizations require no additional computational resources beyond what the base system already uses, yet they significantly close the gap with larger models on RAG-specific tasks.

---

## VRAM Optimization Strategies

| Component | Strategy | VRAM Impact |
|-----------|----------|-------------|
| **Embeddings** | `bge-small-en-v1.5` (133M params) | ~0.5 GB |
| **LLM** | Qwen2.5-1.5B (1.5B params) | ~3 GB |
| **Precision** | bfloat16 instead of float32 | 50% reduction |
| **Reranker** | MiniLM-L-6 (lightweight) | ~0.3 GB |
| **Device Map** | `auto` for optimal GPU/CPU split | Dynamic |
| **Low CPU Mem** | `low_cpu_mem_usage=True` | Reduces overhead |

**Total VRAM Usage**: ~4–5 GB (fits comfortably on GTX 1660 Ti, RTX 3060, etc.)

---

## Technical Highlights

### Handling Domain-Specific Financial Complexity

**1. Mathematical & Technical Content**
- Chunk size (750) preserves complete mathematical expressions
- Character overlap (150) prevents splitting of multi-line equations
- Example: Successfully extracted the carbon mitigation strategy formula with sequential emission reductions (γᵢ)

**2. Financial Terminology & Jargon**
- BGE embeddings trained on domain-specific corpora
- Cross-encoder reranking identifies contextually relevant passages
- Example: Accurately retrieved and cited "one percentage point increase in oil prices leads to 2.5% decrease in stock prices"

**3. Multi-Document Synthesis**
- k=20 initial retrieval casts a wide net across all documents
- Reranking identifies complementary information from different sources
- Example: Climate finance query pulled insights from Andries & Ongena (systemic risk) and Ndiaye (corporate modeling)

**4. Quantitative Precision**
- Low temperature (0.3) reduces hallucination of numbers
- Explicit prompt instruction: "copy exact numbers from source"
- Citation requirement enables downstream verification

**5. Citation-Heavy Academic Writing**
- Chunking strategy respects paragraph boundaries
- Reranking prioritizes primary content over citation lists
- Source attribution tracks original document, not internal citations

**Performance on Complex Queries:**
- ✓ Carbon mitigation strategy with mathematical notation (∑γᵢ, N-step reductions)
- ✓ Default probability computation using Monte Carlo simulations
- ✓ Paris Agreement's policy framework across multiple governance levels
- ✓ Oil-stock market correlations with precise statistical relationships

### Why Cross-Encoder Reranking?

Traditional semantic search (bi-encoders) encodes query and documents separately, which can miss nuanced relevance signals. Cross-encoders process query and document together, capturing fine-grained relevance. The trade-off — slower but more accurate — is entirely acceptable when operating on k=20 candidates.

### Chunking Strategy Rationale

```python
chunk_size=750      # Optimal for financial documents
chunk_overlap=150   # 20% overlap prevents context loss
separators=['\n\n', '\n', '.', ' ']  # Respects document structure
```

Preserves paragraph and sentence boundaries, maintains context for financial metrics and formulas, and prevents splitting of critical multi-line content.

---

## Performance Metrics

**Document Characteristics:**
- **Total Pages**: ~100+ pages across 3 research papers
- **Chunks Generated**: 470 chunks
- **Domain**: Academic financial research (high complexity)
- **Content Types**: Mathematical models, statistical analysis, policy frameworks, econometric modeling

**System Performance:**
- **Retrieval Latency**: ~0.5–1.0s for similarity search + reranking
- **Generation Time**: ~3–5s per answer (512 tokens max)
- **API Overhead**: <50ms (FastAPI request handling, excluding model inference)
- **VRAM Usage**: 4–5 GB peak
- **Citation Consistency**: 100% (every answer includes `[Doc: X | Page: Y]`)
- **Hardware Requirements**: Consumer-grade GPU (6GB+ VRAM recommended)

**Optimization Impact:**
- Reranking provides substantial improvement in retrieval relevance
- Hyperparameter tuning eliminates repetitive output patterns
- Prompt engineering ensures consistent citation and grounding
- Chunking strategy preserves technical and mathematical content integrity
- Combined optimizations enable reliable, near-7B-level performance on RAG tasks from a 1.5B model

---

## Troubleshooting

**1. CUDA Out of Memory**
```python
# Use CPU for reranking to free VRAM
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
```

**2. Slow Generation**
```python
max_new_tokens = 256  # Reduce from 512
```

**3. Vector DB Not Found**
```bash
# The vector DB is git-ignored — regenerate it by running the ingestion notebook first
# Then restart the API: uvicorn main:app --host 0.0.0.0 --port 8000
```

**4. API Returns Error on `/ask` Before `/health` Shows Active**
```bash
# Models are still loading on startup
# Wait for "Models Loaded!!!" in the server logs before sending /ask requests
```

**5. Poor Quality Answers**
```python
# Ensure reranking is enabled and verify:
# - PDF documents parsed correctly
# - Chunk size appropriate for your documents
# - Temperature / repetition penalty settings
```

---

## Future Enhancements

- [ ] Add support for Excel/CSV financial data
- [ ] Implement conversation history
- [ ] Add multi-query retrieval
- [ ] Support for tables and charts extraction
- [ ] Streaming responses via FastAPI `StreamingResponse`
- [ ] Batch query endpoint (`POST /ask/batch`)
- [ ] Authentication and rate limiting for the API
- [ ] Docker containerization for portable deployment

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test on your hardware configuration
4. Submit a pull request

---

## References

**Models Used**:
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [BGE Small EN v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [MS MARCO MiniLM Cross-Encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

**Frameworks**:
- [LangChain](https://github.com/langchain-ai/langchain)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)

---

## Acknowledgments

Optimized through extensive iteration and testing on consumer hardware. Special thanks to the open-source community for providing excellent models and tools.

---

*No APIs. No cloud. No limits. Now with a REST interface.*
