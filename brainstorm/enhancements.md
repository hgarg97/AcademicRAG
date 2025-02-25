Great! Here are some enhancements you can add to your RAG-based chatbot for better citations and usability:

---

## 1️⃣ **Highlight Exact Sentences from Papers**

Instead of just displaying the chunk, you can **highlight** the most relevant sentence(s) from each retrieved chunk.

### ✅ **How to Do It:**

- Use **BM25** (from `rank_bm25`) or a simple **TF-IDF** similarity to rank sentences within each chunk.
- Extract the top-matching sentence(s) for better citation.

### 🔧 **Implementation**

```python
from rank_bm25 import BM25Okapi
import nltk

def extract_relevant_sentences(query, chunks):
    nltk.download("punkt")
    relevant_sentences = []

    for chunk in chunks:
        sentences = nltk.sent_tokenize(chunk)
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        bm25 = BM25Okapi(tokenized_sentences)

        query_tokens = nltk.word_tokenize(query)
        scores = bm25.get_scores(query_tokens)

        best_sentence = sentences[scores.argmax()]
        relevant_sentences.append(best_sentence)

    return relevant_sentences
```

💡 **Modify your retrieval logic** to return these extracted sentences.

---

## 2️⃣ **Inline Citations in LLM Response**

Instead of showing references separately, you can **inject citations** (e.g., `[Broderick et al., 2010]`) **directly into the AI's response**.

### ✅ **How to Do It:**

- Track which chunks contribute to the response.
- Append paper names as references **inline** in the generated answer.

### 🔧 **Implementation**

Modify `generate_answer()`:

```python
def generate_answer(user_query, context_chunks, paper_sources):
    context_text = "\n\n".join(context_chunks)
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LLM_MODEL
    raw_answer = response_chain.invoke({"user_query": user_query, "document_context": context_text})

    # Add citations inline
    citations = {paper: idx + 1 for idx, paper in enumerate(set(paper_sources))}
    formatted_citations = " ".join([f"[{idx}] {paper}" for paper, idx in citations.items()])
    return f"{raw_answer}\n\n**References:** {formatted_citations}"
```

🔹 **Example Output:**

> The omasal sampling technique is widely used for quantifying nitrogen metabolism in cattle [1].  
> **References:**  
> [1] 10-Broderick-2010.pdf

---

## 3️⃣ **Clickable PDF Links**

If you have local PDFs or hosted URLs, you can **display a clickable reference** in Streamlit.

### ✅ **How to Do It:**

- Store **file paths** or **URLs** for papers.
- Generate links in Streamlit.

### 🔧 **Implementation**

Modify the expander section:

```python
with st.expander("🔍 Retrieved Context Chunks (Click to Expand)"):
    for idx, (chunk, paper) in enumerate(zip(relevant_chunks, paper_sources)):
        st.markdown(f"**Chunk {idx + 1}:**")
        st.info(chunk)

        # Add PDF link if available
        paper_path = f"papers/{paper}"  # Adjust the path accordingly
        if os.path.exists(paper_path):  # Local PDF
            st.markdown(f"[📄 Open Paper]({paper_path})", unsafe_allow_html=True)
```

🔹 **Example in UI:**  
📄 **Open Paper** (clickable link)

---

## 4️⃣ **Auto-Generated Bibliography**

If you want **formatted citations (APA, IEEE, etc.)**, use `scholarly` or `manubot`:

```python
from manubot.cite import citeproc

def format_citation(paper_name):
    citation = citeproc.cite(paper_name, format="apa")  # Other formats: ieee, mla
    return citation
```

🔹 Example:

> Broderick, G. A. (2010). _Quantifying ruminal nitrogen metabolism using the omasal sampling technique in cattle—A meta-analysis_. **Journal of Dairy Science**.

---

### 🚀 **Final Thoughts**

All these features can **greatly improve** the credibility and usability of your chatbot. Let me know which one(s) you want to implement first! 😊
