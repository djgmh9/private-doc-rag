import getpass
import os
import os
from dotenv import load_dotenv
from marshmallow import pprint


load_dotenv()


# select a chat model
from langchain.chat_models import init_chat_model


model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# select an embedding model

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# select a vector store
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

from google import genai
from langsmith import wrappers


def main():
    # genai.Client() reads GOOGLE_API_KEY / GEMINI_API_KEY from the environment
    gemini_client = genai.Client()

    # Wrap the Gemini client to enable LangSmith tracing
    client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "python"],
            "metadata": {
                "integration": "google-genai",
            },
        },
    )

    # Make a traced Gemini call
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents="Explain quantum computing in simple terms.",
    )

    print(response.text)


if __name__ == "__main__":
    main()