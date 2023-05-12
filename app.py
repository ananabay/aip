import streamlit as st
from PIL import Image
import math, torch 

from preprocess_data import mask_entities
MASK = '<MASK>'
from database import check_input_in_db, enter_to_db
from load_checkpoints import load_model


author_loaded = load_model('author')
dialect_loaded = load_model('dialect')
period_loaded = load_model('period')

# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.sep.join(dir_path.split(os.path.sep)[:-1]))

text = ""

def initialize_session_vars(vars):
    for key, val in vars.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_vars({
    'text' : text,
    'result' : '',
    'author': '',
    'year': '',
    'book title': '',
    'dialect': ''
})

no_match_text = ''

def predict(text):
    predictions = {}

    for loaded in [author_loaded, dialect_loaded, period_loaded]:

        tokens = loaded[1].encode_plus(
                text,
                padding="max_length",
                truncation=True,
                return_tensors="pt"  # Returns PyTorch tensors
                )
        outputs = loaded[0](**tokens)
        pred = torch.argmax(outputs.logits)
        prediction = loaded[2][int(pred)]

        predictions[loaded[3]] = prediction
    
    return predictions 

def update_displayed():
    text = st.session_state['text']
    # preprocess their input: mask proper names
    masked_input = mask_entities(text, MASK)
    # results = <look up text in db>
    result = check_input_in_db(masked_input)
    # display result author w/ 100% confidence
    if result:
        display_string = 'Your quote showed up in these entries\n\n'
        for entry in result:
            display_string += f"\n\tAuthor: {entry[3]}" +\
                f"\n\tBook: {entry[2]}" +\
                f"\n\tTime period: {entry[5]}" +\
                f"\n\tDialect: {entry[4]}\n\n"
        st.session_state['result'] = display_string
    else:
        predictions = predict(text)
        no_match_text = f'Your quote matches no entry in our database.\n\n{predictions}\n\nTo create a new entry, '+\
                        'fill out the fields below and click submit:'
        st.session_state['result'] = no_match_text


st.set_page_config(layout="wide")

def update_text_from_file():
    if st.session_state['uploaded_file']:
        st.session_state['uploaded_text'] = st.session_state['uploaded_file'].getvalue().decode()
    st.session_state['text'] = st.session_state['uploaded_text']

def update_text_from_input():
    st.session_state['text'] = st.session_state['input_text']

colH, colIm = st.columns([10,3])

with colH:
    st.header('Author Identification Project\n(AIP 🐵)')

with colIm:
    image = Image.open('monkeys.png')
    st.image(image, caption='Our ananabay team members!')

col1, col2 = st.columns([11, 5])

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

col3, col6, col4, col5 = st.columns([4, 10, 2, 2])

col12, col13 = st.columns([1, 1])
# columns for entering new entry into database

with col3:
    st.button('Submit', on_click=update_displayed)
with col4:
    st.button('snow ❄️', on_click=st.snow)
with col5:
    st.button('fun 🎈', on_click=st.balloons)

st.markdown(st.session_state["result"])

def add_entry_to_db():
    text = st.session_state['text']
    masked_input = mask_entities(text, MASK)
    year = int(st.session_state['year'])
    start = math.floor(year/50)*50
    end = (math.floor(year/50)+1)*50 if (math.floor(year/50)+1)*50<2023 else "present"
    period = f'{start}-{end}'
    enter_to_db(masked_input, st.session_state['book'], st.session_state['author'], st.session_state['dialect'], period)

container = st.empty()
if 'matches no entry in our database' in st.session_state['result']:
    with container:
        col8, col9, col10, col11, col12 = st.columns([3, 3, 3, 3, 3])
        with col8:
            author_input = st.text_input('author', key='author')
        with col9:
            book_input = st.text_input('book', key='book')
        with col10:
            year_input = st.text_input('year', key='year')
        with col11:
            dialect_input = st.text_input('dialect', key='dialect')
        with col12:
            submit_new_entry_button = st.button('submit', on_click=add_entry_to_db)
