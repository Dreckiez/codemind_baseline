import sys
from unittest.mock import MagicMock

# ragas's LLM module optionally imports langchain_community's ChatVertexAI for
# VertexAI support. That submodule was removed in current langchain_community
# (0.4.x) versions, so ragas fails to import at all unless something fills the
# gap. We're not using VertexAI, so mocking it out is the correct workaround
# here, not just a hack to suppress an error.
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.vertexai'] = MagicMock()

import json
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

# 1. Load the OpenRouter Key from environment
# RAGAS uses an LLM as a judge to grade your pipeline outputs
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set (check your .env file)")

# Setup the Judge LLM via OpenRouter, wrapped so ragas's metric classes can use it
judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-oss-20b:free",
        temperature=0,
        timeout=180.0,
        max_retries=5
    )
)

judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        check_embedding_ctx_length=False,
        encoding_format="float",
    )
)

# 2. Load your baseline results
with open("baseline_antigravity_results.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

with open("ground_truths.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)

if len(raw_data) != len(gt_data):
    raise ValueError(
        f"baseline_vector_results.json has {len(raw_data)} rows but "
        f"ground_truths.json has {len(gt_data)} rows — they must line up 1:1."
    )

# 3. Format data into the current ragas schema:
# user_input / response / retrieved_contexts / reference
rows = []
for item, gt in zip(raw_data, gt_data):
    rows.append(
        {
            "user_input": item["question"],
            "response": item["answer"],
            "retrieved_contexts": item["contexts"],
            "reference": gt["ground_truth"],
        }
    )

dataset = EvaluationDataset.from_list(rows)

# 4. Run Evaluation
print("Running RAGAS evaluation metrics...")
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ],
    run_config=RunConfig(max_workers=1, max_retries=15, max_wait=60)
)

# 5. Output Results
print("\n=== Baseline RAGAS Scores ===")
print(result)

# Save to a spreadsheet format for your report
df = result.to_pandas()
df.to_csv("ragas_baseline_antigravity_scores.csv", index=False)
print("\nDetailed scores saved to ragas_baseline_antigravity_scores.csv")