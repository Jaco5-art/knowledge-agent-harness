from __future__ import annotations

import os

from .provider import OpenAIProvider
from .rag import FAISSKnowledgeBase, OpenAIEmbeddingProvider
from .workflow import run_workflow


def analyze(document: str, task: str, top_k: int):
    if not document.strip():
        return {"error": "Enter a document before running the workflow."}, []
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "Set OPENAI_API_KEY in the environment before launching the API demo."}, []
    knowledge_base = FAISSKnowledgeBase(OpenAIEmbeddingProvider())
    knowledge_base.add_documents({"user-document": document})
    state = run_workflow(document, task, OpenAIProvider(), knowledge_base=knowledge_base)
    verified = [fact.model_dump(mode="json") for fact in state.get("verified_facts", [])]
    rejected = [fact.model_dump(mode="json") for fact in state.get("rejected_facts", [])]
    summary = {
        "plan": state.get("plan", []),
        "correction_rounds": state.get("correction_round", 0),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "verified_facts": verified,
    }
    return summary, rejected


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('Install the demo extra: pip install -e ".[demo]"') from exc
    with gr.Blocks(title="Multi-Agent Knowledge Verification") as demo:
        gr.Markdown("# Multi-Agent Knowledge Extraction and Verification")
        gr.Markdown("Extract structured facts, retrieve source evidence, verify claims and correct unsupported output.")
        document = gr.Textbox(lines=12, label="Source document")
        task = gr.Textbox(value="extract entities, events and relationships", label="Task")
        top_k = gr.Slider(1, 5, value=3, step=1, label="Retrieved chunks per fact")
        run_button = gr.Button("Run verified extraction", variant="primary")
        verified_output = gr.JSON(label="Verified output")
        rejected_output = gr.JSON(label="Rejected or unresolved facts")
        run_button.click(analyze, [document, task, top_k], [verified_output, rejected_output])
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()

