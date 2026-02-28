# grpc_server.py
import logging
import time
from concurrent import futures

import grpc

from src.recommendation_bert_api.generated import recommendation_pb2
from src.recommendation_bert_api.generated import recommendation_pb2_grpc

from src.recommendation_bert_api.routes_utils import (
    _get_recommendation_explanation,
    _get_user_recommendation_explanation,
)

import torch
from sentence_transformers import util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationServicer(recommendation_pb2_grpc.RecommendationServiceServicer):
    """gRPC сервер, переиспользующий логику из FastAPI app.state"""

    def __init__(self, app):
        self.app = app

    # ==================== SearchQuests ====================
    def SearchQuests(self, request, context):
        start_time = time.time()
        try:
            if not self.app.state.quest_embeddings:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Сначала добавьте квесты")
                return recommendation_pb2.SearchQuestsResponse()

            query_embedding = self.app.state.model.encode(
                request.query,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            results = []
            for quest_id, quest_embedding in self.app.state.quest_embeddings.items():
                if request.category:
                    quest_data = self.app.state.quests_data.get(quest_id)
                    if quest_data and quest_data.get('category') != request.category:
                        continue
                score = util.cos_sim(query_embedding, quest_embedding).item()
                if score > 0.1:
                    quest_data = self.app.state.quests_data.get(quest_id, {})
                    results.append({
                        **quest_data,
                        "similarity_score": float(score),
                        "id": quest_id
                    })

            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            search_time = (time.time() - start_time) * 1000
            top_k = request.top_k if request.top_k > 0 else 5

            proto_results = []
            for r in results[:top_k]:
                proto_results.append(recommendation_pb2.SearchQuestResult(
                    id=r.get("id", 0),
                    title=r.get("title", ""),
                    description=r.get("description", ""),
                    category=r.get("category", ""),
                    similarity_score=r.get("similarity_score", 0.0),
                ))

            return recommendation_pb2.SearchQuestsResponse(
                results=proto_results,
                query_embedding_size=query_embedding.shape[0],
                search_time_ms=round(search_time, 2),
            )

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return recommendation_pb2.SearchQuestsResponse()

    # ==================== HealthCheck ====================
    def HealthCheck(self, request, context):
        return recommendation_pb2.HealthCheckResponse(
            status="healthy",
            model="paraphrase-multilingual-MiniLM-L12-v2",
            device=self.app.state.device,
            quests_count=len(self.app.state.quest_embeddings),
        )


def serve_grpc(app, port=50051):
    """Запуск gRPC-сервера в отдельном потоке"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recommendation_pb2_grpc.add_RecommendationServiceServicer_to_server(
        RecommendationServicer(app), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"gRPC server started on port {port}")
    return server
