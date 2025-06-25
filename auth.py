import streamlit as st
import bcrypt
import logging

# Set up logging
logger = logging.getLogger(__name__)

# User database
USERS = {
    "admin": "$2b$12$yFEU4Hj3Dm.YtBwjYbVl0.dfRSRGc6MS2IItRg6.vj2rAKq15D9Cy",  # password: admin123
    "user": "$2b$12$OkHTg3.QUC/CJApXbkX1G.bFjAZCPSsAfvC1J950hYxVGtZkOIVUO"   # password: user123
}

def check_password(username, password):
    # Verifies user credentials against stored password hash
    if username in USERS:
        return bcrypt.checkpw(password.encode('utf-8'), USERS[username].encode('utf-8'))
    return False

def login_form():
    # Displays and handles the login form interface
    st.sidebar.title("🔐 Login")
    
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if check_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                logger.info(f"User {username} logged in successfully")
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
                logger.warning(f"Failed login attempt for username: {username}")

def logout():
    # Handles user logout functionality
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        logger.info("User logged out")
        st.rerun()

def is_authenticated():
    # Checks if a user is currently authenticated
    return st.session_state.get('authenticated', False)

def get_current_user():
    # Retrieves the username of the currently logged-in user
    return st.session_state.get('username', None) 