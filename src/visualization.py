import plotly.express as px
import plotly.graph_objects as go
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import pandas as pd


def plotly_comparison(
    targets: list[torch.tensor],
    predictions: list[torch.tensor],
    images_dir: Path,
) -> None:
    fig = px.scatter(x=[t.item() for t in targets], y=[p.item() for p in predictions])
    fig.update_layout(yaxis_range=[0, 1])
    os.makedirs(images_dir, exist_ok=True)
    fig.write_image(images_dir / "target_v_pred.png")
    return None


def plotly_losses(
    train_losses: list[float],
    val_losses: list[float],
    images_dir: Path,
    num_epochs: int,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=train_losses, mode="lines+markers", name="train"))
    fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=val_losses, mode="lines+markers", name="validation"))
    os.makedirs(images_dir, exist_ok=True)
    fig.write_image(images_dir / "losses.png")
    return None


def make_confusion_matrix_plotly(y_test, y_pred, classes: list, file_path: Path) -> go.Figure:
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    fig = ff.create_annotated_heatmap(
        z=conf_matrix_normalized,
        x=classes,
        y=classes,
        annotation_text=conf_matrix.astype(str),
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title_text="Confusion Matrix of Sentiment Classification",
        xaxis_title="Predicted Labels",
        yaxis_title="True Labels",
        xaxis=dict(tickmode="array", tickvals=list(range(len(classes))), ticktext=classes),
        yaxis=dict(tickmode="array", tickvals=list(range(len(classes))), ticktext=classes),
    )
    fig_dir = os.path.split(file_path)[0]
    os.makedirs(fig_dir, exist_ok=True)
    fig.write_html(file_path)
    
    return fig
