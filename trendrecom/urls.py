from django.urls import path
from .views import TopTrendingAPIView

urlpatterns = [
    path("recommendations/", TopTrendingAPIView.as_view(), name="top-trending"),
]
