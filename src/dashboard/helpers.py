import plotly.graph_objects as go
import plotly.express as px
import requests


writing_color = "white"
bg_color = "#02024d"

colors_dict = {
    "writing": {"light": "#02024d", "dark": "white"},
    "bg": {"light": "white", "dark": "#02024d"},
}


def apply_layout(fig: go.Figure, sublib: str, mode: str = "light") -> go.Figure:
    writing_color = colors_dict["writing"][mode]
    bg_color = colors_dict["bg"][mode]
    if sublib == "go":
        fig.update_layout(
            title=dict(font=dict(size=18, color=writing_color), x=0.5),
            legend=dict(font=dict(size=14, color=writing_color)),
            xaxis=dict(
                title=dict(font=dict(size=14, color=writing_color)),
                tickfont=dict(color=writing_color)
            ),
            yaxis=dict(
                title=dict(font=dict(size=14, color=writing_color)),
                tickfont=dict(color=writing_color)
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor='rgba(0, 0, 0, 0)',
        )
    elif sublib == "px":
        fig.update_layout(
            title=dict(font=dict(size=18, color=writing_color), x=0.5),
            legend=dict(font=dict(size=14, color=writing_color)),
            xaxis=dict(title=dict(font=dict(size=14, color=writing_color))),
            yaxis=dict(title=dict(font=dict(size=14, color=writing_color))),
            xaxis_tickfont=dict(color=writing_color),
            yaxis_tickfont=dict(color=writing_color),
            paper_bgcolor=bg_color,
            plot_bgcolor='rgba(0, 0, 0, 0)',
        )
    else:
        raise ValueError(f"Sublib must be one of 'go', 'px'. Received: {sublib}")
    return fig


def get_route_result(url: str) -> list[dict]:
    try:
        response = requests.post(url)
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred: {e}")
    
    return data
