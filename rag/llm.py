import os
from typing import List
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    openai = None

def generate_answer(question: str, context: str) -> str:
    if openai and openai.api_key:
        # minimal call (OpenAI or Azure OpenAI if envs are set)
        resp = openai.chat.completions.create(
            model=os.getenv("CHAT_MODEL","gpt-4o-mini"),
            messages=[{"role":"system","content":"Answer using only the context. Cite sources ids if relevant."},
                      {"role":"user","content": f"Context:\n{context}\n\nQuestion: {question}"}],
            temperature=0
        )
        return resp.choices[0].message.content
    # no key: simple extractive fallback (top context returned)
    return f"Context-based answer:\n{context[:800]}"
