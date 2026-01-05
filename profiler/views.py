from django.shortcuts import render
from .ml_model import get_player_list, get_player_profile


def player_list_view(request):
    players_df = get_player_list()
    players = players_df.to_dict(orient="records")

    query = request.GET.get("q", "")
    if query:
        q = query.lower()
        filtered = []
        for p in players:
            name = str(p.get("player_name_ko", "")).lower()
            team = str(p.get("team_name_ko", "")).lower()
            if q in name or q in team:
                filtered.append(p)
        players = filtered

    context = {
        "players": players,
        "query": query,
    }
    return render(request, "profiler/player_list.html", context)


def player_detail_view(request, player_id):
    profile = get_player_profile(player_id)
    if profile is None:
        return render(request, "profiler/not_found.html", status=404)

    context = {
        "profile": profile,
    }
    return render(request, "profiler/player_detail.html", context)