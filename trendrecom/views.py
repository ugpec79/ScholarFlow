import pandas as pd
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class TopTrendingAPIView(APIView):
    def get(self, request):
        file_path = "trending_papers.csv"

        if not os.path.exists(file_path):
            return Response(
                {
                    "error": "Trending data not found. Please wait for initial generation."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            trending = pd.read_csv(file_path)
            top_n = int(request.query_params.get("limit", 10))
            top_papers = trending.head(top_n).to_dict(orient="records")
            return Response(top_papers, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
