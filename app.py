import sys, os
import streamlit as st
from PIL import Image

from preprocess_data import mask_entities
MASK = '<MASK>'
from database import check_input_in_db, enter_to_db

# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.sep.join(dir_path.split(os.path.sep)[:-1]))

text = "This is some example text with Yangyang and Ruth and Mr. Spongebob Squarepants and a 4th person in Boston on April 25th."

def initialize_session_vars(vars):
    for key, val in vars.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_vars({
    'text' : text,
    'result' : ''
})

def update_displayed():
    text = st.session_state['text']

    # preprocess their input: mask proper names
    masked_input = mask_entities(text, MASK)
    # results = <look up text in db>
    result = check_input_in_db(masked_input)

    # display result author w/ 100% confidence
    if result:
        display_string = ''
        for entry in result:
            display_string += f"\nYour quote showed up in these entries\nAuthor: {entry[3]};\n" +\
                f"Book: {entry[2]};" +\
                f"Time period: {entry[5]};" +\
                f"Dialect: {entry[4]}\n\n"
        st.session_state['result'] = display_string
    # else:
    #       get results from BERT
    #       display result + distribution
    #       ask the user if they want to enter the string into the db
    #       if yes,
    #           let them input author, date, dialect – with the model's top predictions as default
    #           preprocess their input: mask proper names + normalize author name + year -> period
    #           enter result into database
    

def update_text_from_file():
    if st.session_state['uploaded_file']:
        st.session_state['uploaded_text'] = st.session_state['uploaded_file'].getvalue().decode()
    st.session_state['text'] = st.session_state['uploaded_text']

def update_text_from_input():
    st.session_state['text'] = st.session_state['input_text']

colH, colIm = st.columns([8,5])

with colH:
    st.header('Author Identification Project\n(AIP)')

with colIm:
    image = Image.open('monkeys.png')
    st.image(image, caption='Our ananabay team members!')


col1, col2 = st.columns([8, 6])

with col1:
    st.text_area('Input some text:', 
                value=text, 
                on_change=update_text_from_input, 
                key='input_text',
                height=120)

with col2: 
    st.file_uploader('Or upload a file:', 
                    type=['.txt'], 
                    on_change=update_text_from_file, 
                    key='uploaded_file',
                    )

col3, col4, col5 = st.columns([11, 3, 3])

with col3:
    st.button('Submit', on_click=update_displayed)
with col4:
    st.button('snow ❄️', on_click=st.snow)
with col5:
    st.button('fun 🎈', on_click=st.balloons)

st.markdown(st.session_state["result"])
