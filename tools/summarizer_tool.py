# """A simple final summarization tool – keeps OpenAI call in one place."""
# from langchain import OpenAI

# from ..config import OPENAI_MODEL

# def summarize_answer(text: str) -> str:
#     """
#     Post-process full recipe answer into a shorter 3-sentence summary.
#     """
#     llm = OpenAI(model_name=OPENAI_MODEL, temperature=0.3)
#     prompt = (f"将下列菜谱答案压缩成不超过 3 句的摘要，保持关键信息：\n\n{text}")
#     return llm(prompt)
