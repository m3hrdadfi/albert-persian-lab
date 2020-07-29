import streamlit as st
import pandas as pd
from PIL import Image


def load_image(image_path, image_resize=None):
    """ A helper function to load and resize an image

    Args:
        image_path (str): The image path.
        image_resize (tuple): The image-resize width, height

    Returns:
        return Pillow.Image object
    """

    image = Image.open(image_path)
    if isinstance(image_resize, tuple):
        image.resize(image_resize)
    return image


def load_css(css_path):
    """ Load css locally.

    Args:
        css_path (str): The css path.
    """
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(css_url):
    """ Load css remotely

    Args:
        css_url (str): The css url address.
    """
    st.markdown(f'<link href="{css_url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(_st, icon_name, link=None, class_name=''):
    if link:
        _st.markdown(f'<a href="{link}" class="{class_name}"><i class="fa {icon_name}"></i></a>',
                     unsafe_allow_html=True)
    else:
        _st.markdown(f'<i class="fa {icon_name}"></i>', unsafe_allow_html=True)


def icon_group(_st, icon_names, links):
    """ Put icons into a group

    Args:
        _st (object): The streamlit instantiation
        icon_names (list:str): List of icons' names
        links (list:str): List of icons' hyperlink

    Returns:

    """
    md = '<div class="nav-icons">'
    for name, link in zip(icon_names, links):
        md += f'<a href={link}><i class="fa {name}"></i></a>'

    md += '</div>'

    _st.markdown(md, unsafe_allow_html=True)


def load_snippet(snippet_path, language='python'):
    """ Load the snippet like `Syntaxhighlighter`

    Args:
        snippet_path (str): The snippet file path.
        language (str): The snippet language

    """

    with open(snippet_path) as f:
        st.code(f'{f.read()}', language=language)


def task_configuration(config_path):
    """ NLP-Task configuration/mapping """
    df = pd.read_json(config_path)
    names = df.name.values.tolist()

    mapping = {
        df['name'].iloc[i]: (
            df['text'].iloc[i],
            df['labels'].iloc[i],
            df['description'].iloc[i],
            df['model_name'].iloc[i],
            df['direction'].iloc[i],
            df['mapper'].iloc[i],
        ) for i in range(len(names))}

    return names, mapping
