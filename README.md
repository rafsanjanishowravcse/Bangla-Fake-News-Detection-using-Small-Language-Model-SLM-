# Bangla Fake News Detection with SLM

This project implements an intelligent system capable of verifying whether a news claim — provided as text or audio — is **REAL**, **FAKE**, or **UNSURE**.

It uses a hybrid architecture that combines:
*   **Retrieval-Augmented Generation (RAG)** for grounding the model's reasoning in factual web-based evidence.
*   **SarvamAI APIs** for real-time speech-to-text (STT) conversion and translation.
*   **ChatGroq-powered LLMs** (Qwen2.5, Mistral, LLaMA) for fact-checking, explanation, and natural language reasoning.

## Key Features

*   **Input Flexibility**: Accepts user input in multiple languages and formats (typed text, spoken audio).
*   **Multilingual Processing**: Automatically translates non-English claims to English using SarvamAI and processes them through the LLM pipeline.
*   **Dual-Language Output**: Explanations are returned in both English and the original input language for clarity and accessibility.
*   **Grounded Fact Checking**: Verifies claims using real-time search results (trusted sources e.g., news18.com, ptinews.com) and generates evidence-supported explanations via a LangChain RAG workflow.
*   **Powered by Open Source LLMs**: Supports Qwen2.5, Mistral, and Phi-3-mini through the ChatGroq LLM API for fast and scalable responses.

## Project Structure

```
.
├── code/               # Source code for the application
│   ├── app.py          # Main application entry point
│   └── ...
├── dataset/            # Datasets used for evaluation
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Saiful-Islam0/Bangla-Fake-News-Detection-SLM.git
    cd Bangla-Fake-News-Detection-SLM
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration:**
    Create a `.env` file in the `code/` directory (or root, depending on where you run it) with the following keys:
    ```env
    SARVAM_API_KEY=your_key_here
    GROK_API_KEY=your_key_here
    SERP_DEV_API_KEY=your_key_here
    model_multi_query=...
    model_summarizer=...
    model_judge=...
    ```

4.  **Run the Application:**
    Navigate to the `code` directory and run the app:
    ```bash
    cd code
    python app.py
    ```

## How It Works

1.  Accepts a news claim as text or audio (in any Indian language).
2.  Translates to English using SarvamAI (if needed).
3.  Performs intelligent web search using Serper.dev.
4.  Applies multi-query RAG to gather and summarize evidence.
5.  Uses an LLM to classify the claim as REAL / FAKE / UNSURE with explanation.
6.  Optionally translates the verdict back to the original language.

## Evaluation Metrics

The following tables summarize performance across languages and input types using different LLMs and strategies.

### Strategies
*   **Strategy 1**: Multi-query generation -> Document retrieval -> Summarization -> Verdict.
*   **Strategy 2**: Multi-query generation -> Document retrieval -> Raw documents -> Verdict.
*   **Strategy 3**: Direct retrieval (original claim) -> Summarization -> Verdict.

### LLM Evaluation — English Claims (Strategy 3)

| Model | TC | F1R | F1F |
| :--- | :--- | :--- | :--- |
| llama3-8b-8192 | 0.64 | 0.88 | 0.71 |
| qwen/qwen3-32b | 0.72 | 0.9 | 0.74 |
| mistral-saba-24b | 0.67 | 0.9 | 0.68 |
| deepseek-r1-distill-llama-70b | 0.69 | 0.9 | 0.69 |
| meta-llama/llama-4-scout-17b-16e-instruct | 0.7 | 0.91 | 0.71 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 0.52 | 0.95 | 0.65 |
| qwen-qwq-32b | 0.67 | 0.91 | 0.73 |

*(TC - TotalCoverage, F1R - F1 Score(Real), F1F - F1 Score(Fake))*

### LLM Evaluation — Regional Languages (Hindi/Kannada)

| Model | Coverage | F1 Score (Real) | F1 Score (Fake) |
| :--- | :--- | :--- | :--- |
| llama3-8b-8192 | 0.68 | 0.92 | 0.69 |
| qwen/qwen3-32b | 0.75 | 0.91 | 0.76 |
| mistral-saba-24b | 0.68 | 0.90 | 0.60 |
| deepseek-r1-distill-llama-70b | 0.70 | 0.88 | 0.58 |
| meta-llama/llama-4-scout-17b-16e-instruct | 0.70 | 0.89 | 0.55 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 0.66 | 0.89 | 0.55 |
| qwen-qwq-32b | 0.66 | 0.89 | 0.57 |

### LLM Evaluation — Audio Inputs (Multilingual)

| Model | Coverage | F1 Score (Real) | F1 Score (Fake) |
| :--- | :--- | :--- | :--- |
| llama3-8b-8192 | 0.22 | 0.57 | 0.28 |
| mistral-saba-24b | 0.22 | 0.95 | 0.52 |
| qwen-qwq-32b | 0.51 | 0.66 | 0.64 |

### SarvamAI Speech & Translation Performance

| Metric | Score |
| :--- | :--- |
| WER | 0.2887 |
| CER | 0.0887 |
| BLEU Score | 0.2027 |
| METEOR Score | 0.4949 |
| BERTScore | 0.9149 |

## Contributors
Saiful Islam
