from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('diagnosis/', views.diagnosis, name='diagnosis'),
    path('reversal/', views.reversal, name='reversal'),
]