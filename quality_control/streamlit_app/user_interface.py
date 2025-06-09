import streamlit as st
import json
import os

def fetch_data(path):
    try:
        absolute_path = os.path.abspath(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"This file does not exist {path}: {e}")
        return {}


def cached_fetch_data(path):
    return fetch_data(path)

def format_passage_data(data):
    passage_data = [
        (entry["passage_id"], entry["passage"], entry["qa"])
        for entry in data
    ]
    return passage_data, len(passage_data)

@st.cache_data
def cached_format_passage_data(data):
    return format_passage_data(data)

if 'current_passage' not in st.session_state:
    st.session_state['current_passage'] = 0

class UserInterface:
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_ratings = cached_fetch_data(f"submissions/{user_id}.json")
        self.data = cached_fetch_data(f"subsets/{user_id}.json")
        self.passage_data, self.data_size = cached_format_passage_data(self.data)

    def render_sidebar(self):
        with st.sidebar:
            st.markdown('<style>.tomato-title { color: #FF6347; }</style>', unsafe_allow_html=True)
            st.markdown('<h1 class="tomato-title">Rating Criteria</h1>', unsafe_allow_html=True)
            st.markdown("""
                - **⭐ Factual**: The answer should be factually correct and accurate.
                - **⭐ Grounded in Passage**: The answer should be fully supported by the passage text.
                - **⭐ Relevant**: The Q&A should refer to medically relevant content. It is irrelevant if it contains specific details about a study's experimental design, statistical analysis methods, tables or figures, study dates, locations, funding sources, or other details that are not essential for understanding the key medical knowledge.
            """, unsafe_allow_html=True)

    def render(self):
        self.render_sidebar()

        unrated_passage_index = next(
            (i for i, entry in enumerate(self.passage_data) if entry[0] not in self.user_ratings), None
        )

        if unrated_passage_index is None:
            st.success("You have completed rating all passages. Thank you for your participation!")
            return  # Exit the function to avoid displaying further passages


        # Set the current passage to the next unrated passage
        st.session_state['current_passage'] = unrated_passage_index
        progress = st.session_state['current_passage'] / self.data_size
        st.progress(progress)

        current_entry = self.passage_data[st.session_state['current_passage']]
        passage_id, passage, qa = current_entry

        if passage_id not in self.user_ratings:
            self.user_ratings[passage_id] = {}

        self.display_data(qa, passage, passage_id)

    def display_data(self, pairs, passage_text, passage_id):
        with st.form(key=f'qa_pair_selection_for_{passage_id}'):
            st.markdown(
                "<style>.info-text { color: #ff6347; font-weight: bold; font-size: 18px; }</style>"
                "<div class='info-text'>Please rate the Q&A below based on the criteria on the left and the passage text below: </div>",
                unsafe_allow_html=True
            )
            for i, (qa_id, q, a) in enumerate(pairs, 1):
                st.markdown(
                    f"<style>.heading {{ color: #ff6347; font-weight: bold; font-size: 20px; }}</style>"
                    f"<div class='heading' style='margin-top: 20px;'>Question {i}:</div>"
                    f"<div>{q}</div><div class='heading'>Answer {i}:</div><div>{a}</div>",
                    unsafe_allow_html=True
                )

                criteria = ["Factual", "Grounded in Passage", "Relevant"]
                user_choices = [criterion for criterion in criteria if st.checkbox(criterion, key=f'checkbox_{criterion}_{passage_id}_{i}')]

                self.user_ratings[passage_id][i] = user_choices

            st.markdown(
                f"<style>.heading {{ color: #ff6347; font-weight: bold; font-size: 20px; }}</style>"
                f"<div class='heading' style='margin-top: 20px;'>Passage Text:</div><div>{passage_text}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<style>.stButton>button { height: 3em; width: 7em; font-size: 1em; font-weight: bold; }</style>",
                unsafe_allow_html=True
            )
            if st.form_submit_button(label='NEXT'):
                self.handle_submission()

    def handle_submission(self):
        path = f"submissions/{self.user_id}.json"
        try:
            with open(path, 'w') as f:
                json.dump(self.user_ratings, f)
            st.session_state['current_passage'] += 1
            if st.session_state['current_passage'] < self.data_size:
                st.rerun()
            else:
                st.write("You have completed rating all passages.")
        except Exception as e:
            st.error(f"Failed to save data: {e}")
