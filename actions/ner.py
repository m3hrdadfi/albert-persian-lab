import streamlit as st
import time

from utils import is_identical, plot_result
from utils.loader import task_configuration, local_css, load_snippet
from utils.model import load_config, load_tokenizer, load_model, token_predicting


def do_ner(task_title, task_config_filename, do_print_code=False):
    st.title(task_title)

    config_names, config_map = task_configuration('assets/%s.json' % task_config_filename)
    example = st.selectbox('Choose an example', config_names)
    # st.markdown(config_map[example][2], unsafe_allow_html=True)

    height = min((len(config_map[example][0].split()) + 1) * 2, 200)
    if config_map[example][4] == 'rtl':
        local_css('assets/rtl.css')

    sequence = st.text_area('Text', config_map[example][0], key='sequence', height=height)
    labels = st.text_input('Labels (comma-separated)', config_map[example][1], max_chars=1000)
    original_labels = config_map[example][1].split(', ')

    labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))

    if len(labels) == 0 or len(sequence) == 0:
        st.write('Enter some text and at least one label to see predictions.')
        return

    if not is_identical(labels, original_labels, 'list'):
        st.write('Your labels must be as same as the NLP task `%s`' % task_title)
        return

    if st.button('Analyze'):
        if do_print_code:
            load_snippet('snippets/named_entity_recognition_code.txt', 'python')

        s = st.info('Predicting ...')
        model_config = load_config(config_map[example][3])
        labels = model_config.id2label
        mapper = config_map[example][5]
        # labels = {k: labels_mapper[v] for k, v in model_config.id2label.items()}

        tokenizer = load_tokenizer(config_map[example][3])
        model = load_model(config_map[example][3], 'TFAlbertForTokenClassification')
        predictions = token_predicting(model, tokenizer, sequence, labels)

        ner_color = []
        ner_tags = []
        for tag, translate in mapper.items():
            if tag != 'O':
                if translate[0] not in ner_tags:
                    ner_color.append(
                        '<span style="background-color: %s;">%s</span>'
                        % (translate[1], translate[0]))
                    ner_tags.append(translate[0])

        ner_color = ' '.join(ner_color)

        ner_sequence = []
        for token, label in predictions:
            if token not in ['[CLS]', '[SEP]']:
                label = mapper[label]
                if label[0]:
                    ner_sequence.append(
                        '<span style="background-color: %s;" class="token token-ner">%s <span class="ner-label">%s</span></span>'
                        % (label[1], token, label[0]))
                else:
                    ner_sequence.append(
                        '<span class="token">%s</span>'
                        % token)

        time.sleep(1)
        s.empty()

        st.markdown(f'<p class="ner-color">{ner_color}</p>', unsafe_allow_html=True)
        ner_sequence = ' '.join(ner_sequence)

        st.markdown(f'<p class="ner-box">{ner_sequence}</p>', unsafe_allow_html=True)
