from django.urls import path
from . import views

urlpatterns = [
    # 루트('/')로 들어오면 선수 리스트 보여주기
    path("", views.player_list_view, name="player_list"),

    # 특정 선수 상세
    path("players/<int:player_id>/", views.player_detail_view,
         name="player_detail"),
]