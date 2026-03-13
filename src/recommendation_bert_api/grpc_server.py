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

        # ==================== AddUsers ====================
    def AddUsers(self, request, context):
        try:
            from internal.pydantic_models.pydantic_models import User

            storage = self.app.state.storage

            for user_proto in request.users:
                user_id = user_proto.user_id
                quest_ids = list(user_proto.quest_ids)

                if user_id in self.app.state.users_data:
                    existing_user = self.app.state.users_data[user_id]
                    existing_quest_ids = existing_user.get("quest_ids", [])
                    if sorted(existing_quest_ids) == sorted(quest_ids):
                        logger.info(f"Пользователь {user_id} уже существует с такими же quest_ids, пропускаем")
                        continue
                    else:
                        logger.info(f"Пользователь {user_id} существует, но quest_ids изменились. Обновляем.")

                self.app.state.users_data[user_id] = {
                    "user_id": user_id,
                    "quest_ids": quest_ids
                }

                if quest_ids:
                    user_embeddings = []
                    valid_quests = []

                    for quest_id in quest_ids:
                        if quest_id in self.app.state.quest_embeddings:
                            user_embeddings.append(self.app.state.quest_embeddings[quest_id])
                            valid_quests.append(quest_id)

                    if len(valid_quests) != len(quest_ids):
                        self.app.state.users_data[user_id]["quest_ids"] = valid_quests
                        logger.warning(f"У пользователя {user_id} найдено {len(valid_quests)} из {len(quest_ids)} квестов")

                    if len(user_embeddings) == 0:
                        logger.warning(f"У пользователя {user_id} нет валидных эмбеддингов квестов")
                        continue

                    try:
                        user_embeddings_tensor = torch.stack(user_embeddings)
                        user_profile_embedding = torch.mean(user_embeddings_tensor, dim=0)
                        self.app.state.profile_embeddings[user_id] = user_profile_embedding

                        user = User(user_id=user_id, quest_ids=quest_ids)
                        storage.save_user(user, user_profile_embedding)

                    except Exception as e:
                        logger.error(f"Ошибка создания профиля для пользователя {user_id}: {e}")
                        context.set_code(grpc.StatusCode.INTERNAL)
                        context.set_details(str(e))
                        return recommendation_pb2.AddUsersResponse()

            return recommendation_pb2.AddUsersResponse(
                status="success",
                total_users=len(self.app.state.users_data),
                total_profiles=len(self.app.state.profile_embeddings),
                message=f"Обработано {len(request.users)} пользователей"
            )

        except Exception as e:
            logger.error(f"Ошибка добавления пользователя: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return recommendation_pb2.AddUsersResponse()


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
