import streamlit as st
import numpy as np
import plotly.express as px
import collections


def is_identical(a, b, kind='list'):
    if kind == 'list':
        return collections.Counter(a) == collections.Counter(b)

    return False


def plot_result(labels, scores, predicted):
    scores *= 100
    fig = px.bar(x=scores, y=labels, orientation='h',
                 labels={'x': 'Confidence', 'y': 'Label'},
                 text=scores,
                 range_x=(0, 115),
                 title=f'Predicted `{predicted}`',
                 color=np.linspace(0, 1, len(scores)),
                 color_continuous_scale='Viridis')
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    st.plotly_chart(fig)
