
from fastai.vision import load_learner, open_image
from PIL import Image
import streamlit as st
import warnings
warnings.filterwarnings(action = 'ignore')

model_path = '../Movie Poster to Genre/posters/'
st.title('Movie Poster to Genre Classifier')
uploaded_file = st.file_uploader("Upload a Movie Poster...", type=["jpg", "jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)
learn = load_learner(model_path)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = open_image(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    pred_class, pred_idx, outputs = learn.predict(img)
    labels = pred_class.obj

    st.write(f'The labels are \n {labels}')
    