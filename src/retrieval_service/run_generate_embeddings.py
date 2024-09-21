# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os

from langchain.embeddings import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

async def main() -> None:
    # Connect to MongoDB
    mongo_client = MongoClient(os.environ.get("ATLAS_URI"))
    collection = mongo_client["GeminiRAG"]["chat-rag"]
    # Use VertexAI Embeddings Model to embed vector into MongoDB Atlas Vector Search
    embed_service = VertexAIEmbeddings(model_name="textembedding-gecko@001")
    # Load the PDF File
    loader = PyPDFLoader('../data/Cover_Letter.pdf')
    # Split the PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(loader)
    # Store the chunks into MongoDB Atlas Vector Search
    vector_store = MongoDBAtlasVectorSearch(collection, embed_service)
    vector_store.add_documents(split_docs)
    
    print("Imported the document into the MongoDB Atlas vector store.")

if __name__ == "__main__":
    asyncio.run(main())
