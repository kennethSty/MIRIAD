from user_interface import UserInterface
import json
import streamlit as st
import os
from streamlit_authenticator import Authenticate
import yaml
from yaml.loader import SafeLoader
import os

@st.cache_data
def get_config():
    with open('auth_config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    return config


config = get_config()

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

@st.cache_resource
def get_ui(username):
    return UserInterface(username)


def main():


    # Page initialization
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
        st.session_state['authentication_status'] = None

    # Login page
    if st.session_state['page'] == 'login':
        # Assuming the authenticator.login function returns (name, authentication_status, username)
        name, authentication_status, username = authenticator.login('main')
        st.session_state['username'] = username
        st.session_state['authentication_status'] = authentication_status

        if st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')

        # Check if authentication was successful
        if st.session_state["authentication_status"]:
            if 'username' in st.session_state:
                username = st.session_state['username']
                ui = UserInterface(f"user_{username}")
                #ui = get_ui(f"user_{username}")
                ui.render()



if __name__ == "__main__":
    main()
