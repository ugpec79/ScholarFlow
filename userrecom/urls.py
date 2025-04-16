from django.urls import path
from .views import RecommendationAPIView

urlpatterns = [
    path('recommend/<str:user_id>/', RecommendationAPIView.as_view(), name='recommendation-api'),
]
