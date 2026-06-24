import ollama
from loguru import logger


class ReviewSummarizer:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        logger.info(f"Summarizer initialized with model: {model}")

    def _build_prompt(self, query: str, results: list[dict]) -> str:
        if not results:
            return "No results found to summarize."

        reviews_text = "\n\n".join(
            [
                f"Review {i+1} ProductID: {r['product_id']} (Rating: {r['rating']}/5, Sentiment: {r['bert_sentiment']}):\n{r['text'][:300]}"
                for i, r in enumerate(results)
            ]
        )

        return f"""You are a helpful assistant that summarizes food product reviews.

        The user searched for: "{query}"

        Here are the top search results:

        {reviews_text}

        Provide a concise summary (3-5 sentences) of what people are saying about "{query}".
        Highlight the most common opinions, and overall sentiment.
        Do not mention review numbers. Just summarize naturally.
        
        Highlight the top 3 products given their product IDs.
        
        """

    def stream(self, query: str, results: list[dict]):
        """Generator that yields text chunks for streaming."""
        if not results:
            yield "No results found to summarize."
            return

        prompt = self._build_prompt(query, results)
        stream = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]
