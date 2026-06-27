
# RAG & Agent Evaluation Pipeline

This repository contains the automated evaluation framework used to benchmark and compare two distinct retrieval and generation baselines against a human-verified Ground Truth dataset derived from the `flask-login` repository.

Evaluation metrics are programmatically computed using the **RAGAS (Retrieval Augmented Generation Assessment)** framework with LLM-as-a-judge infrastructure.

---

## 1. Repository Architecture

```text
├── ground_truths.json               # 10 human-verified Q&A pairs + golden context
├── baseline_vector_results.json     # Baseline 1: Raw inputs, outputs, and retrieved chunks
├── baseline_antigravity_results.json # Baseline 2: Antigravity IDE inputs, outputs, and context arrays
├── baseline_ragas.py                # Main RAGAS evaluation script
├── ragas_baseline_scores.csv        # Baseline 1: Row-by-row and average RAGAS scores
└── ragas_baseline_antigravity_scores.csv # Baseline 2: Row-by-row and average RAGAS scores
```

---

## 2. Prerequisites & Setup

### Environment Variables

To prevent credential leaks, API keys are isolated within a local `.env` file. Create a file named `.env` in the root directory and add your OpenRouter credentials:

```env
OPENROUTER_API_KEY=your_actual_openrouter_api_key_here
```

### Installation

Install the required dependencies using `pip`:

```bash
pip install python-dotenv ragas langchain-openai langchain-community langchain-text-splitters langchain-chroma langchain-classic langchain-core
```

---

## 3. Execution Walkthrough

### Step 1: Baseline Data Generation

The evaluation engine expects raw inference outputs to follow a strict schema matching RAGAS inputs. Both baseline stages compile their results into individual JSON files structured as an array of objects:

```json
[
  {
    "question": "The user query...",
    "answer": "The model's generated response...",
    "contexts": ["Code chunk 1 string...", "Code chunk 2 string..."]
  }
]

```

* **Baseline 1 (Standard Vector RAG):** Generated programmatically using LangChain, ChromaDB, and `nvidia/llama-nemotron-embed-vl-1b-v2:free`. Results are written directly to `baseline_vector_results.json`.
* **Baseline 2 (Google Antigravity IDE):** Generated inside the agentic IDE environment. A meta-prompt forces the autonomous agent to format its final answers and code boundaries directly into a JSON file, saved as `baseline_antigravity_results.json`.

### Step 2: Running the Evaluation Script

The `baseline_ragas.py` script automatically maps the baseline logs against `ground_truths.json`, routes requests securely through OpenRouter to the evaluation models, handles rate limits via 180-second connection timeouts, and outputs a formatted CSV.

**Before running the script, you MUST manually update the file paths inside `baseline_ragas.py` depending on which baseline you want to evaluate.** If you do not change the output file name, you will overwrite existing data.

1. Open `baseline_ragas.py`.
2. Find the input section and point it to the correct baseline JSON:
```python
# Change this to either "baseline_vector_results.json" OR "baseline_antigravity_results.json"
with open("baseline_vector_results.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
```

3. Scroll to the very bottom of the script and update the CSV export name:
```python
# Change this to match your baseline so you do not overwrite previous tests!
df.to_csv("ragas_baseline_scores.csv", index=False) 
```

Once the files are correctly mapped, execute the script:

```bash
python baseline_ragas.py
```

---

## 4. Understanding the Outputs

### Reading the CSV Files

The evaluation script generates two metrics sheets: `ragas_baseline_scores.csv` and `ragas_baseline_antigravity_scores.csv`.

Each row in the CSV represents one of the 10 Ground Truth questions, providing individual performance marks scored from `0.0000` (Failure) to `1.0000` (Perfect Match).

#### Columns Definitions

* **`user_input`**: The actual question being asked (e.g., *"Which specific Python class implements the custom login view?"*).
* **`retrieved_contexts`**: The array of raw text chunks that your vector database or Antigravity IDE explicitly fetched to read before answering.
* **`response`**: The final generated text answer that your pipeline outputted.
* **`reference`**: Human-verified "golden" answer. The LLM judge uses this as the ultimate source of truth to grade the `response` against.
* **`faithfulness`:** Evaluates factual grounding. Measures whether the generation model stuck strictly to the code boundaries present in the context window, flagging ungrounded claims or external model hallucinations.
* **`answer_relevancy`:** Evaluates conversational accuracy. Measures how directly the generated text aligns with the prompt's explicit goal, flagging extraneous fluff or off-topic explanations.
* **`llm_context_precision_with_reference`:** Evaluates retrieval sorting and signal-to-noise ratio. Measures whether highly relevant code chunks were correctly prioritized at the top of the context window, minimizing distracting code.
* **`context_recall`:** Evaluates retrieval completeness. Measures whether the system successfully fetched *all* underlying codebase blocks required by the ground truth key to answer the prompt.
