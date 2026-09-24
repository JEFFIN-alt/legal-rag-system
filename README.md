# LEGAL RAG

## Evidence-Grounded Legal RAG System

LEGAL RAG is a **Retrieval-Augmented Generation (RAG) system for legal document analysis and question answering**.

It combines semantic search, BM25 lexical retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, context expansion, and Gemini-based generation to produce answers grounded in retrieved legal documents.

The current knowledge base contains the **Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA)**.

---

## 🎯 Project Objective

Large language models can generate fluent answers but may produce unsupported or fabricated information.

LEGAL RAG addresses this problem by retrieving relevant passages from a controlled legal document collection and supplying those passages to the language model as evidence.

The system follows the principle:

> **Retrieve first. Generate second.**

Instead of asking the LLM to answer from its general knowledge, LEGAL RAG provides retrieved document evidence and instructs the model to ground its response in that evidence.

---

## 🧠 System Architecture

```text
                    LEGAL DOCUMENTS
                 BNS / BNSS / BSA
                         │
                         ▼
                PDF Text Extraction
                         │
                         ▼
                  Text Cleaning
                         │
                         ▼
              Continuous Chunking
                         │
                         ▼
                 Sentence Embeddings
                         │
                         ▼
                    ChromaDB
                         │
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     Semantic Retrieval          BM25 Retrieval
            │                         │
            └────────────┬────────────┘
                         ▼
                  RRF Fusion
                   Top 50
                         │
                         ▼
               Cross-Encoder
                  Reranking
                   Top 5
                         │
                         ▼
               Context Expansion
                         │
                         ▼
                  Gemini LLM
                         │
                         ▼
              Grounded Legal Answer
                         │
                         ▼
              Evidence + Citations
```

---

## 🔎 Retrieval Pipeline

LEGAL RAG uses a multi-stage retrieval architecture.

### 1. Semantic Retrieval

Legal document chunks are converted into vector embeddings using:

```text
all-MiniLM-L6-v2
```

The embeddings are stored in ChromaDB and used for semantic similarity search.

### 2. BM25 Retrieval

A BM25 lexical retriever searches the complete legal corpus using keyword-based matching.

This helps retrieve passages containing important exact terms such as:

```text
Section 35
arrest without warrant
bail
electronic record
attempt to murder
```

### 3. Reciprocal Rank Fusion

Semantic and BM25 rankings are combined using **Reciprocal Rank Fusion (RRF)**.

The current pipeline retrieves:

```text
Top 50 candidates
```

before reranking.

### 4. Cross-Encoder Reranking

The retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The current pipeline selects the top:

```text
5 evidence chunks
```

### 5. Context Expansion

Neighboring chunks are added around the highest-ranked result to provide additional surrounding context.

### 6. Grounded Generation

The retrieved evidence is supplied to Gemini.

The model is instructed to:

* use only supplied evidence
* avoid unsupported claims
* distinguish law from inference
* cite the supplied document pages
* explicitly state when evidence is insufficient

---

## 📚 Current Legal Corpus

The current repository contains documents corresponding to:

| Document                                        | Purpose                  |
| ----------------------------------------------- | ------------------------ |
| Bharatiya Nyaya Sanhita (BNS), 2023             | Substantive criminal law |
| Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 | Criminal procedure       |
| Bharatiya Sakshya Adhiniyam (BSA), 2023         | Law of evidence          |

The current processed corpus contains:

```text
Total chunks: 1,000
```

The generated processed JSON files and ChromaDB database are intentionally excluded from Git and can be regenerated locally.

---

## 🧪 Example

### Question

```text
What is the charge for attempt to murder?
```

### LEGAL RAG response structure

```text
ANSWER
...

LAW SAYS
...

CASE DOCUMENT SAYS
...

INFERENCE
...

SOURCES
...
```

The system successfully retrieved relevant BNS provisions concerning attempt to murder and generated a source-grounded response using the retrieved evidence.

---

## 🛠️ Technology Stack

### Programming

* Python

### Document Processing

* PyMuPDF

### Retrieval

* ChromaDB
* Sentence Transformers
* BM25
* Reciprocal Rank Fusion

### Reranking

* Cross-Encoder
* `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Generation

* Google Gemini
* Gemini Interactions API

### Environment

* Conda
* Python 3.12

---

## 📁 Project Structure

```text
legal-rag-system/
│
├── data/
│   └── documents/
│       ├── BNS2023.pdf
│       ├── 250884_2_english_01042024.pdf
│       └── 250882_english_01042024_0.pdf
│
├── src/
│   ├── __init__.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── indexer.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── hybrid_retrieve.py
│   ├── reranker.py
│   ├── context_builder.py
│   ├── generator.py
│   ├── pipeline.py
│   ├── retrieve.py
│   ├── search.py
│   ├── evaluate_retrieval.py
│   └── evaluate_reranking.py
│
├── data/processed/
│   └── Generated chunk files
│
├── chroma_db/
│   └── Local vector database
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/JEFFIN-alt/legal-rag-system.git
cd legal-rag-system
```

Create the Conda environment:

```bash
conda create -n ai-lab python=3.12
conda activate ai-lab
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Gemini API Key

LEGAL RAG requires a Gemini API key for answer generation.

Set the key as an environment variable:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Verify that it is available:

```bash
python -c "import os; print(bool(os.getenv('GEMINI_API_KEY')))"
```

The API key should **never be committed to GitHub**.

---

## ▶️ Running LEGAL RAG

### Step 1 — Process documents

Run the document processing pipeline:

```bash
python src/chunker.py
```

This extracts, cleans, and chunks the PDF documents.

### Step 2 — Build the vector index

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/indexer.py
```

This generates embeddings and stores them in ChromaDB.

### Step 3 — Run LEGAL RAG

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.pipeline
```

The system will prompt:

```text
Ask LEGAL RAG (type 'exit' to quit):
```

Enter a legal question and LEGAL RAG will retrieve, rerank, expand, and generate an evidence-grounded response.

---

## 📊 Current Validation

The RAG pipeline has been tested through its complete execution path:

```text
PDF
 ↓
Chunking
 ↓
Embedding
 ↓
Vector Index
 ↓
Semantic Retrieval
 ↓
BM25 Retrieval
 ↓
RRF
 ↓
Cross-Encoder Reranking
 ↓
Context Expansion
 ↓
Gemini Generation
```

A retrieval test over the legal corpus successfully identified relevant BNSS material for questions concerning arrest procedures.

A subsequent end-to-end test successfully retrieved BNS material concerning **attempt to murder** and generated a structured response containing:

* Answer
* Law Says
* Case Document Says
* Inference
* Sources

---

## 🚧 Current Limitations

LEGAL RAG is currently a **research prototype / MVP**.

Current limitations include:

* The corpus is currently limited to three principal legal documents.
* Case-specific document ingestion is not yet implemented.
* The current reranker is a general passage-ranking model rather than an Indian-law-specific reranker.
* Context expansion currently focuses on the highest-ranked retrieved result.
* Retrieval evaluation is currently limited and should be expanded with a larger manually verified benchmark.
* Legal provisions should be verified against the applicable official and current version before real-world use.

---

## 🔮 Future Work

Planned extensions include:

### Case Document Analysis

Support documents such as:

```text
FIRs
Complaints
Charge Sheets
Court Orders
Judgments
Witness Statements
Police Reports
Contracts
Evidence Documents
```

### Legal Knowledge Expansion

Extend the knowledge base with:

```text
Constitution of India
Central and State Acts
Rules and Regulations
Court Judgments
Legal Commentaries
```

### Improved Legal Retrieval

Future versions can explore:

* legal-domain embedding models
* legal-specific rerankers
* section-aware retrieval
* citation-aware retrieval
* metadata filtering
* document versioning

### User Interface

A web interface can provide:

* legal document upload
* conversational querying
* source inspection
* page-level citations
* evidence highlighting
* case-document comparison

---

## ⚖️ Disclaimer

LEGAL RAG is an **academic/research project intended for legal information retrieval and decision-support experimentation**.

It is not a substitute for professional legal advice, legal representation, or judicial determination.

Users should independently verify legal information against authoritative and current legal sources before relying on it.

---

## 👨‍💻 Author

**JEFFIN-alt**

B.Tech — Artificial Intelligence & Data Science

GitHub:

https://github.com/JEFFIN-alt

---

## 📄 License

License information will be added in a future version.
::[Open LEGAL RAG on GitHub](https://github.com/JEFFIN-alt/legal-rag-system)
