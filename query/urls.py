from django.urls import path
from .views import HybridSearchView

urlpatterns = [
    path("hybrid/", HybridSearchView.as_view(), name="hybrid_search"),
]