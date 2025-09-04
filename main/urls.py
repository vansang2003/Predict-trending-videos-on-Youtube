from django.urls import path
from . import views

app_name = 'main'

urlpatterns = [
    path('', views.index, name='index'),
    path('video/<int:video_id>/', views.video_detail, name='video_detail'),
    path('kenh/', views.channel_list, name='channel_list'),
    path('du-lieu/', views.data, name='data'),
    path('du-doan/', views.predict, name='predict'),
    path('tai-khoan/', views.account, name='account'),
] 