import streamlit as st

from utils.loader import local_css, remote_css
from utils.loader import icon_group
from utils.loader import load_image
from actions.clf_sa import do_sentiment
from actions.clf_tc import do_text_classification
from actions.ner import do_ner
from actions.g_tg import do_text_generation

MODEL_DESC = {
    'ALBERT-Persian Sentiment Analysis': """""",
    'ALBERT-Persian Text Classification': """""",
    'ALBERT-Persian NER': """""",
    'ALBERT-Persian Text Generation': """""",
}
APP_DESC = """"""
SIDEBAR_FOOTER = """[ALBERT-Persian](https://github.com/m3hrdadfi/albert-persian) is the first attempt on ALBERT for the Persian Language. The model was trained based on [Google's ALBERT BASE Version 2.0](https://github.com/google-research/albert) over various writing styles from numerous subjects (e.g., scientific, novels, news) with more than 3.9M documents, 73M sentences, and 1.3B words, like the way we did for [ParsBERT](https://github.com/m3hrdadfi/albert-persian)."""

do_print_code = False
logo_size = 120
# logo = load_image('assets/logo-2x250.png', image_resize=(logo_size, logo_size))


def main():
    global do_print_code
    remote_css("https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font/dist/font-face.css")
    local_css('assets/style.css')
    remote_css("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css")

    # Sidebar
    # st.sidebar.image(logo, width=logo_size)
    icon_group(st.sidebar,
               [
                   'fa-github',
                   'fa-twitter',
                   'fa-linkedin',
               ], [
                   'https://github.com/m3hrdadfi',
                   'https://twitter.com/m3hrdadfi',
                   'https://www.linkedin.com/in/m3hrdadfi/',
               ])
    st.sidebar.markdown('<h1 class="text-center">ALBERT Persian Lab</h1>', unsafe_allow_html=True)
    st.sidebar.markdown(APP_DESC, unsafe_allow_html=True)

    model_desc = st.sidebar.selectbox('Model', list(MODEL_DESC.keys()), 0)
    do_print_code = st.sidebar.checkbox('Show code snippet', False)
    st.sidebar.markdown('#### Model Description')
    st.sidebar.markdown(MODEL_DESC[model_desc], unsafe_allow_html=True)
    st.sidebar.markdown(SIDEBAR_FOOTER)

    # Main
    if model_desc == list(MODEL_DESC.keys())[0]:
        do_sentiment(
            list(MODEL_DESC.keys())[0],
            task_config_filename='albert_sentiment_analysis',
            do_print_code=do_print_code)
    elif model_desc == list(MODEL_DESC.keys())[1]:
        do_text_classification(
            list(MODEL_DESC.keys())[1],
            task_config_filename='albert_text_classification',
            do_print_code=do_print_code)
    elif model_desc == list(MODEL_DESC.keys())[2]:
        do_ner(
            list(MODEL_DESC.keys())[2],
            task_config_filename='albert_named_entity_recognition',
            do_print_code=do_print_code)
    elif model_desc == list(MODEL_DESC.keys())[3]:
        do_text_generation(
            list(MODEL_DESC.keys())[3],
            task_config_filename='albert_text_generation',
            do_print_code=do_print_code)
    else:
        do_sentiment(
            list(MODEL_DESC.keys())[0],
            task_config_filename='albert_sentiment_analysis',
            do_print_code=do_print_code)


if __name__ == "__main__":
    main()
