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

from contextlib import asynccontextmanager
from ipaddress import IPv4Address, IPv6Address

import os
from fastapi import FastAPI
from langchain.embeddings import VertexAIEmbeddings
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

from .routes import routes

class AppConfig(BaseModel):
    host: IPv4Address | IPv6Address = IPv4Address("127.0.0.1")
    port: int = 8080
    atlas_uri: str
    mongodb_db: str

def parse_config() -> AppConfig:
    config = {}
    config["host"] = os.environ.get("APP_HOST", "127.0.0.1")
    config["port"] = os.environ.get("APP_PORT", 8080)
    config["atlas_uri"] = os.environ.get("ATLAS_URI", "mongodb://localhost:27017")
    config["mongodb_db"] = os.environ.get("MONGODB_DB", "GeminiRAG")
    return AppConfig(**config)

def gen_init(cfg: AppConfig):
    async def initialize_mongodb(app: FastAPI):
        # Initialize MongoDB connection
        app.state.mongodb_client = AsyncIOMotorClient(cfg.atlas_uri)
        app.state.mongodb_db = app.state.mongodb_client[cfg.mongodb_db]
        app.state.embed_service = VertexAIEmbeddings(model_name="textembedding-gecko@001")
        yield
        # Close MongoDB connection on shutdown
        app.state.mongodb_client.close()

    return asynccontextmanager(initialize_mongodb)

def init_app(cfg: AppConfig) -> FastAPI:
    app = FastAPI(lifespan=gen_init(cfg))
    app.include_router(routes)
    return app
