import asyncio
import json
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import RootModel, SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


USER_ID_CONTEXT_PROMPT = """You find user IDs whose profile descriptions match a hobby-related question.

Use only the user IDs and profile descriptions from the retrieved context. Group matching IDs by the hobby named in the question. Do not create IDs, include user data, or use IDs that are not in the context. If no profiles match, return an empty object.

{format_instructions}"""


class HobbyUserIds(RootModel[dict[str, list[int]]]):
	"""Maps each matched hobby to the matching user IDs."""


def format_profile_document(user: dict[str, Any]) -> Document:
	user_id = int(user["id"])
	about_me = str(user.get("about_me") or "")
	return Document(
		page_content=f"User ID: {user_id}\nAbout me: {about_me}",
		metadata={"user_id": user_id},
	)


class HobbiesSearchWizard:
	def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
		self.embeddings = embeddings
		self.llm_client = llm_client
		self.user_client = UserClient()
		self.vectorstore: Chroma | None = None

	async def __aenter__(self):
		self.vectorstore = Chroma(
			collection_name="user_hobby_profiles",
			embedding_function=self.embeddings,
		)
		await self._synchronize_vectorstore()
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		return None

	async def _synchronize_vectorstore(self) -> None:
		"""Add new profiles and remove deleted profiles before every search."""
		if self.vectorstore is None:
			raise RuntimeError("Vectorstore is not initialized.")

		users = await asyncio.to_thread(self.user_client.get_all_users)
		users_by_id = {str(user["id"]): user for user in users}
		stored = await asyncio.to_thread(self.vectorstore.get)
		stored_ids = set(stored["ids"])
		current_ids = set(users_by_id)

		deleted_ids = list(stored_ids - current_ids)
		if deleted_ids:
			await self.vectorstore.adelete(ids=deleted_ids)

		new_ids = list(current_ids - stored_ids)
		if new_ids:
			documents = [format_profile_document(users_by_id[user_id]) for user_id in new_ids]
			await self.vectorstore.aadd_documents(documents, ids=new_ids)

		print(f"Vectorstore synchronized: {len(new_ids)} added, {len(deleted_ids)} removed.")

	async def retrieve_context(self, query: str, k: int = 25, score: float = 0.1) -> str:
		if self.vectorstore is None:
			raise RuntimeError("Vectorstore is not initialized.")

		await self._synchronize_vectorstore()
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

	async def identify_hobby_user_ids(self, query: str, context: str) -> HobbyUserIds:
		parser = PydanticOutputParser(pydantic_object=HobbyUserIds)
		messages = [
			SystemMessagePromptTemplate.from_template(USER_ID_CONTEXT_PROMPT),
			HumanMessage(
				content=f"## RETRIEVED USER PROFILES:\n{context}\n\n## USER QUESTION:\n{query}"
			),
		]
		prompt = ChatPromptTemplate.from_messages(messages).partial(
			format_instructions=parser.get_format_instructions()
		)
		return await (prompt | self.llm_client | parser).ainvoke({})

	async def output_ground(self, grouped_ids: HobbyUserIds) -> dict[str, list[dict[str, Any]]]:
		"""Verify each generated ID still exists and return its live full profile."""
		unique_ids = list(dict.fromkeys(
			user_id
			for user_ids in grouped_ids.root.values()
			for user_id in user_ids
		))

		async def fetch_user(user_id: int) -> dict[str, Any] | None:
			try:
				return await self.user_client.get_user(user_id)
			except Exception:
				print(f"Skipping unavailable user ID: {user_id}")
				return None

		users = await asyncio.gather(*(fetch_user(user_id) for user_id in unique_ids))
		users_by_id = {
			int(user["id"]): user
			for user in users
			if user is not None and "id" in user
		}
		return {
			hobby: [users_by_id[user_id] for user_id in user_ids if user_id in users_by_id]
			for hobby, user_ids in grouped_ids.root.items()
			if any(user_id in users_by_id for user_id in user_ids)
		}


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

	async with HobbiesSearchWizard(embeddings, llm_client) as wizard:
		print("Hobbies Searching Wizard")
		print("Example: I need people who love to go to mountains")
		while True:
			query = input("\n> ").strip()
			if query.lower() in {"exit", "quit"}:
				break
			if not query:
				continue

			context = await wizard.retrieve_context(query)
			if not context:
				print("No relevant profiles found.")
				continue

			grouped_ids = await wizard.identify_hobby_user_ids(query, context)
			result = await wizard.output_ground(grouped_ids)
			print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	asyncio.run(main())



