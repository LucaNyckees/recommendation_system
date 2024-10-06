import plotly.express as px
import plotly.graph_objects as go
import torch
import os
from pathlib import Path


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
