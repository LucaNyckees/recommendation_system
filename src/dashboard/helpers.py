import plotly.graph_objects as go
import plotly.express as px
import requests
from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL

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
            paper_bgcolor='rgba(0, 0, 0, 0)',
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
            paper_bgcolor='rgba(0, 0, 0, 0)',
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


def generate(k, v):
    match v['type']:
        case 'choice':
            return html.Div([html.P(str(k)+'. '+v['question']), dcc.RadioItems(id={'index': k, 'type': v['type'], 'category':'questionnaire', 'additional':False}, options={i: i for i in v['options']})])
        case 'multi-choice':
            return html.Div([html.P(str(k)+'. '+v['question']), dcc.Checklist(id={'index': k, 'type': v['type'], 'category':'questionnaire', 'additional':False}, options={i: i for i in v['options']})])
        case 'choice+blank':
            return html.Div([html.P(str(k)+'. '+v['question']), dcc.RadioItems(id={'index': k, 'type': v['type'], 'category':'questionnaire', 'additional':False}, options={i: i for i in v['options']}), dcc.Input(id={'index': k, 'type': v['type'], 'category':'questionnaire', 'additional':True}, disabled=True)])
        case 'blank':
            return html.Div([html.P(str(k)+'. '+v['question']), dcc.Input(id={'index': k, 'type': v['type'], 'category':'questionnaire', 'additional':False})])
        case 'essay':
            return html.Div([html.P(str(k)+'. ' + v['question']), dcc.Textarea(id={'index': k, 'type': v['type'], 'category':'questionnaire', 'additional':False})])
        case _:
            return html.Div('Something wrong...')
