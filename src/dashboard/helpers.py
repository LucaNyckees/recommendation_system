import plotly.graph_objects as go
import plotly.express as px


light_writing_color = "white"
dark_paper_bg_color = "#02024d"


def darkmode_layout(fig: go.Figure, sublib: str) -> go.Figure:
    if sublib == "go":
        fig.update_layout(
            title=dict(font=dict(size=18, color=light_writing_color), x=0.5),
            legend=dict(font=dict(size=14, color=light_writing_color)),
            xaxis=dict(
                title=dict(font=dict(size=14, color=light_writing_color)),
                tickfont=dict(color=light_writing_color)
            ),
            yaxis=dict(
                title=dict(font=dict(size=14, color=light_writing_color)),
                tickfont=dict(color=light_writing_color)
            ),
            paper_bgcolor=dark_paper_bg_color,
            plot_bgcolor='rgba(0, 0, 0, 0)',
        )
    elif sublib == "px":
        fig.update_layout(
            title=dict(font=dict(size=18, color=light_writing_color), x=0.5),
            legend=dict(font=dict(size=14, color=light_writing_color)),
            xaxis=dict(title=dict(font=dict(size=14, color=light_writing_color))),
            yaxis=dict(title=dict(font=dict(size=14, color=light_writing_color))),
            xaxis_tickfont=dict(color=light_writing_color),
            yaxis_tickfont=dict(color=light_writing_color),
            paper_bgcolor=dark_paper_bg_color,
            plot_bgcolor='rgba(0, 0, 0, 0)',
        )
    else:
        raise ValueError(f"Sublib must be one of 'go', 'px'. Received: {sublib}")
    return fig