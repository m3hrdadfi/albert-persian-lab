import streamlit as st
import time

from utils import is_identical
from utils.loader import task_configuration, local_css, load_snippet
from utils.model import load_tokenizer, load_model, text_generation


def do_text_generation(task_title, task_config_filename, do_print_code=False):
    st.title(task_title)

    config_names, config_map = task_configuration('assets/%s.json' % task_config_filename)
    example = st.selectbox('Choose an example', config_names)
    # st.markdown(config_map[example][2], unsafe_allow_html=True)

    height = min((len(config_map[example][0].split()) + 1) * 2, 200)
    if config_map[example][4] == 'rtl':
        local_css('assets/rtl.css')

    sequence = st.text_area('Text', config_map[example][0], key='sequence', height=height)
    labels = st.text_input('Mask (placeholder)', config_map[example][1], max_chars=1000)
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
            load_snippet('snippets/text_generation_code.txt', 'python')

        s = st.info('Predicting ...')
        tokenizer = load_tokenizer(config_map[example][3])
        model = load_model(config_map[example][3], 'TFAlbertForMaskedLM', from_pt=True)
        masked_words, words = text_generation(model, tokenizer, sequence)

        new_sequence = []
        for index, word in enumerate(words):
            if index in masked_words:
                masks_sequence = []
                for mi in masked_words[index]:
                    masks_sequence.append(
                        '<span class="masked" style="background-color: %s;">%s</span>' %
                        (mi['color'], mi['token_str'])
                    )

                new_sequence.append(
                    '<span class="token"><span class="masks-start">[</span><span class="token-masks">%s</span><span class="masks-end">]</span></span>' %
                    (''.join(masks_sequence))
                )
            else:
                new_sequence.append(
                    '<span class="token">%s</span>' %
                    word
                )

        new_sequence = ' '.join(new_sequence)
        time.sleep(1)
        s.empty()

        st.markdown(f'<p class="masked-box">{new_sequence}</p>', unsafe_allow_html=True)
