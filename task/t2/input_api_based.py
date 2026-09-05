from enum import StrEnum
import os
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `api_based_grounding.png` to see the flow of app

QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts search parameters from user questions about users.

## Available Search Fields:
- **name**: User's first name (e.g., "John", "Mary")
- **surname**: User's last name (e.g., "Smith", "Johnson") 
- **email**: User's email address (e.g., "john@example.com")

## Instructions:
1. Analyze the user's question and identify what they're looking for
2. Extract specific search values mentioned in the query
3. Map them to the appropriate search fields
4. If multiple search criteria are mentioned, include all of them
5. Only extract explicit values - don't infer or assume values not mentioned

## Examples:
- "Who is John?" → name: "John"
- "Find users with surname Smith" → surname: "Smith" 
- "Look for john@example.com" → email: "john@example.com"
- "Find John Smith" → name: "John", surname: "Smith"
- "I need user emails that filled with hiking" → No clear search parameters (return empty list)

## Response Format:
{format_instructions}
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""


llm_client: AzureChatOpenAI | None = None
user_client = UserClient()


def get_llm_client() -> AzureChatOpenAI:
    """Create the DIAL client only when a model request is made."""
    global llm_client
    if llm_client is None:
        if not API_KEY:
            raise RuntimeError("DIAL_API_KEY is not set. Configure it before running this task.")
        llm_client = AzureChatOpenAI(
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version="",
            azure_deployment=os.getenv("DIAL_MODEL", "gpt-4o"),
        )
    return llm_client


class SearchField(StrEnum):
    NAME = "name"
    SURNAME = "surname"
    EMAIL = "email"


class SearchRequest(BaseModel):
    search_field: SearchField = Field(description="The user-service field to search.")
    search_value: str = Field(description="The explicit name, surname, or email from the question.")


class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(default_factory=list)


def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    """Extract search parameters from user query and retrieve matching users."""
    parser = PydanticOutputParser(pydantic_object=SearchRequests)
    messages = [
        SystemMessagePromptTemplate.from_template(QUERY_ANALYSIS_PROMPT),
        HumanMessage(content=user_question),
    ]
    prompt = ChatPromptTemplate.from_messages(messages).partial(
        format_instructions=parser.get_format_instructions()
    )
    search_requests: SearchRequests = (prompt | get_llm_client() | parser).invoke({})

    if not search_requests.search_request_parameters:
        print("No specific search parameters found!")
        return []

    request_params = {
        request.search_field.value: request.search_value
        for request in search_requests.search_request_parameters
    }
    print(f"Searching users with: {request_params}")
    return user_client.search_users(**request_params)


def format_user_context(users: list[dict[str, Any]]) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(format_value(item) for item in value)
        if isinstance(value, dict):
            return "; ".join(f"{key}: {format_value(item)}" for key, item in value.items())
        return str(value)

    return "\n\n".join(
        "User:\n" + "\n".join(f"  {field}: {format_value(value)}" for field, value in user.items())
        for user in users
    )


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    """Combine user query with retrieved context into a formatted prompt."""
    augmented_prompt = USER_PROMPT.format(context=format_user_context(context), query=user_question)
    print(f"\nAugmented prompt:\n{augmented_prompt}")
    return augmented_prompt


def generate_answer(augmented_prompt: str) -> str:
    """Generate final answer using the augmented prompt."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
    response = get_llm_client().invoke(messages)
    return str(response.content)


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    while True:
        user_question = input("\n> ").strip()
        if user_question.lower() in {"quit", "exit"}:
            break
        if not user_question:
            continue

        context = retrieve_context(user_question)
        if context:
            answer = generate_answer(augment_prompt(user_question, context))
            print(f"\nAnswer:\n{answer}")
        else:
            print("No relevant information found.")


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce