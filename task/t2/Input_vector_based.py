import asyncio
import os
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

SYSTEM_PROMPT = """You are a RAG-powered assistant that answers questions about users.

Use only the information in RAG CONTEXT to answer USER QUESTION. If the context does not contain enough relevant information, say so clearly. Present matching user information clearly and do not invent user details."""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(format_value(item) for item in value)
        if isinstance(value, dict):
            return "; ".join(f"{key}: {format_value(item)}" for key, item in value.items())
        return str(value)

    fields = "\n".join(f"  {field}: {format_value(value)}" for field, value in user.items())
    return f"User:\n{fields}"


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("Loading all users...")
        users = await asyncio.to_thread(UserClient().get_all_users)
        documents = [
            Document(page_content=format_user_document(user), metadata={"user_id": str(user.get("id", ""))})
            for user in users
        ]
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        if not documents:
            raise RuntimeError("The user service returned no users to index.")

        document_batches = [
            documents[index:index + batch_size]
            for index in range(0, len(documents), batch_size)
        ]
        vectorstores = await asyncio.gather(
            *(FAISS.afrom_documents(batch, self.embeddings) for batch in document_batches)
        )

        final_vectorstore = vectorstores[0]
        for vectorstore in vectorstores[1:]:
            final_vectorstore.merge_from(vectorstore)
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore is not initialized.")

        results = await asyncio.to_thread(
            self.vectorstore.similarity_search_with_relevance_scores,
            query,
            k=k,
            score_threshold=score,
        )
        context_parts = []
        for document, relevance_score in results:
            print(f"Relevance score: {relevance_score:.3f}\n{document.page_content}\n")
            context_parts.append(document.page_content)
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
        response = self.llm_client.invoke(messages)
        return str(response.content)


async def main():
    if not API_KEY:
        raise RuntimeError("DIAL_API_KEY is not set. Configure it before running this task.")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        azure_deployment=os.getenv("DIAL_EMBEDDING_MODEL", "text-embedding-3-small-1"),
        dimensions=384,
    )
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        azure_deployment=os.getenv("DIAL_MODEL", "gpt-4o"),
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            if not user_question:
                continue

            context = await rag.retrieve_context(user_question)
            if not context:
                print("No relevant information found.")
                continue

            answer = rag.generate_answer(rag.augment_prompt(user_question, context))
            print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce