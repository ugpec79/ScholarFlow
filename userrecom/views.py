from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .recommender import get_user_recommendations


class RecommendationAPIView(APIView):
    def get(self, request, user_id):
        try:
            recommendations = get_user_recommendations(user_id)
            if not recommendations:
                return Response(
                    {"message": f"No recommendations found for user: {user_id}"},
                    status=status.HTTP_404_NOT_FOUND,
                )
            return Response(
                {"user_id": user_id, "recommendations": recommendations},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
