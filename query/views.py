from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .core import hybrid_search, fetch_similar_results_with_scores, rerank_combined_results, add_query_to_redis

class HybridSearchView(APIView):
    def post(self, request):
        query = request.data.get("query")
        user_id = request.data.get("user_id")
        top_k = int(request.data.get("top_k", 10))

        if not query or not user_id:
            return Response({"error": "query and user_id are required"}, status=400)

        current_results = hybrid_search(query, top_k=top_k)
        past_results = fetch_similar_results_with_scores(user_id, current_query=query, top_k=top_k)
        add_query_to_redis(user_id, query)

        final_results = rerank_combined_results(current_results, past_results, query, alpha=0.7, beta=0.3, top_k=top_k)

        return Response(final_results)
