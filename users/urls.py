from django.urls import path
from .views import GuestRegisterView, LogUserActivityView

urlpatterns = [
    path('register-guest/', GuestRegisterView.as_view(), name='register_guest'),
    path('log_activity/', LogUserActivityView.as_view(), name='log_user_activity'),
]