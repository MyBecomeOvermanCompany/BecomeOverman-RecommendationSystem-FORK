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

        # ==================== RecommendUsers ====================
    def RecommendUsers(self, request, context):
        try:
            import http.client as http_client

            cur_user_id = request.user_id

            if cur_user_id not in self.app.state.users_data:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"user (with user_id={cur_user_id}) not found in data")
                return recommendation_pb2.RecommendUsersResponse()

            if cur_user_id not in self.app.state.profile_embeddings:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"user (with user_id={cur_user_id}) profile_embedding not found in data")
                return recommendation_pb2.RecommendUsersResponse()

            cur_user_profile_embedding = self.app.state.profile_embeddings[cur_user_id]
            cur_user_quests = self.app.state.users_data.get(cur_user_id, {}).get("quest_ids", [])

            results = []

            for user_id, profile_embedding in self.app.state.profile_embeddings.items():
                if user_id == cur_user_id:
                    continue

                score = util.cos_sim(cur_user_profile_embedding, profile_embedding).item()

                if score > 0.2:
                    other_user_quests = self.app.state.users_data.get(user_id, {}).get("quest_ids", [])

                    explanation = _get_user_recommendation_explanation(
                        cur_user_id=cur_user_id,
                        cur_user_quests=cur_user_quests,
                        other_user_id=user_id,
                        other_user_quests=other_user_quests,
                        similarity_score=score,
                        quests_data=self.app.state.quests_data
                    )

                    results.append({
                        "user_id": user_id,
                        "similarity_score": float(score),
                        "explanation": explanation
                    })

            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_k = request.top_k if request.top_k > 0 else 10
            top_k = min(max(top_k, 1), len(results))
            top_k_results = results[:top_k]

            proto_results = []
            for r in top_k_results:
                explanation = r.get("explanation", {})
                details = explanation.get("details", {})

                proto_details = recommendation_pb2.ExplanationDetails(
                    common_quests_count=details.get("common_quests_count", 0),
                    common_quests_ids=details.get("common_quests_ids", []),
                    common_categories=details.get("common_categories", []),
                    user_categories_top=details.get("user_categories_top", []),
                    other_user_categories_top=details.get("other_user_categories_top", []),
                    user_quests_count=details.get("user_quests_count", 0),
                    other_user_quests_count=details.get("other_user_quests_count", 0),
                    similarity_level=details.get("similarity_level", ""),
                )

                proto_explanation = recommendation_pb2.UserExplanation(
                    summary=explanation.get("summary", ""),
                    details=proto_details,
                )

                proto_results.append(recommendation_pb2.RecommendedUser(
                    user_id=r["user_id"],
                    similarity_score=r["similarity_score"],
                    explanation=proto_explanation,
                ))

            return recommendation_pb2.RecommendUsersResponse(
                status="success",
                user_id=cur_user_id,
                results=proto_results,
                total_users_analyzed=len(self.app.state.profile_embeddings) - 1,
            )

        except Exception as e:
            logger.error(f"Ошибка рекомендации пользователей: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return recommendation_pb2.RecommendUsersResponse()

    # ==================== RecommendQuests ====================
    def RecommendQuests(self, request, context):
        try:
            user_quest_ids = list(request.user_quest_ids)

            if not user_quest_ids:
                return recommendation_pb2.RecommendQuestsResponse()

            user_embeddings = []
            for quest_id in user_quest_ids:
                if quest_id in self.app.state.quest_embeddings:
                    user_embeddings.append(self.app.state.quest_embeddings[quest_id])

            if not user_embeddings:
                return recommendation_pb2.RecommendQuestsResponse()

            user_embeddings_tensor = torch.stack(user_embeddings)
            user_profile_embedding = torch.mean(user_embeddings_tensor, dim=0)

            results = []
            for quest_id, quest_embedding in self.app.state.quest_embeddings.items():
                if quest_id in user_quest_ids:
                    continue

                if request.category:
                    quest_data = self.app.state.quests_data.get(quest_id)
                    if quest_data and quest_data.get('category') != request.category:
                        continue

                score = util.cos_sim(user_profile_embedding, quest_embedding).item()

                if score > 0.2:
                    quest_data = self.app.state.quests_data.get(quest_id, {})
                    results.append({
                        **quest_data,
                        "similarity_score": float(score),
                        "id": quest_id
                    })

            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            top_k = request.top_k if request.top_k > 0 else 10

            proto_results = []
            for result in results[:top_k]:
                explanation = _get_recommendation_explanation(
                    result,
                    user_quest_ids,
                    self.app.state.quests_data
                )

                proto_results.append(recommendation_pb2.RecommendedQuest(
                    id=result.get("id", 0),
                    title=result.get("title", ""),
                    description=result.get("description", ""),
                    category=result.get("category", ""),
                    similarity_score=result.get("similarity_score", 0.0),
                    explanation=explanation,
                ))

            profile_info = recommendation_pb2.UserProfileInfo(
                quests_count=len(user_quest_ids),
                embedding_dim=user_profile_embedding.shape[0],
                method="mean_pooling",
            )

            return recommendation_pb2.RecommendQuestsResponse(
                recommendations=proto_results,
                user_profile_info=profile_info,
            )

        except Exception as e:
            logger.error(f"Ошибка рекомендаций квестов: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return recommendation_pb2.RecommendQuestsResponse()

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
