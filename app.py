import cv2
from PIL import Image,ImageEnhance
import numpy as np
import os
import streamlit as st

st.set_page_config(page_title='open-cv',page_icon=':smiley:')

def load_image(img):
    im = Image.open(img)
    return im

def load_classifiers():    
    face_cascade = cv2.CascadeClassifier('face.xml')
    eye_cascade = cv2.CascadeClassifier('eyes.xml')
    return face_cascade,eye_cascade

def detect_faces(image):
    new_image=np.array(image.convert('RGB'))
    #cv2.cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV. We will use some of color space conversion codes below.
    img=cv2.cvtColor(new_image,1)
    gray=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 1)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img,faces

def detect_eyes(image):
    new_image=np.array(image.convert('RGB'))
    #cv2.cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV. We will use some of color space conversion codes below.
    img=cv2.cvtColor(new_image,1)
    gray=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.5, 1)
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img,eyes

def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Edges
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    #Color
    color = cv2.bilateralFilter(img, 9, 300, 300)
    #Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon

if __name__ == '__main__':
    face_cascade,eye_cascade=load_classifiers()
    st.title('Open-Cv Guide')
    activities=['Faces','Eyes','Cartonize']
    choice=st.sidebar.selectbox("Select Activity",activities)
    image_file=st.file_uploader('Upload an image',type=['jpg','png','jpeg'])
    if(choice=='Faces'):
        if(image_file):
            img=load_image(image_file)
            placeholder=st.image(img)
        if(image_file and st.sidebar.button('Detect')):
            placeholder.empty()
            face_im,cord=detect_faces(img)
            st.image(face_im)
        click=st.sidebar.button('Generate Code')
        if(click):
            st.code('''
    def detect_faces(image):
        new_image=np.array(image.convert('RGB'))
        #cv2.cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV. We will use some of color space conversion codes below.
        img=cv2.cvtColor(new_image,1)
        gray=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 1)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        return img,faces
            ''')
            

    if(choice=='Eyes'):
        if(image_file):
            img=load_image(image_file)
            placeholder=st.image(img)
        if(image_file and st.sidebar.button('Detect')):
            placeholder.empty()
            eye_im,cord=detect_eyes(img)
            st.image(eye_im)
        click=st.sidebar.button('Generate Code')
        if(click):
            st.code('''
    def detect_eyes(image):
        new_image=np.array(image.convert('RGB'))
        #cv2.cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV. We will use some of color space conversion codes below.
        img=cv2.cvtColor(new_image,1)
        gray=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.5, 1)
        for (x,y,w,h) in eyes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        return img,eyes
            ''')
    if(choice=='cartonize'):
        if(image_file):
            img=load_image(image_file)
            placeholder=st.image(img)
        if(image_file and st.sidebar.button("Cartonize")):
            placeholder.empty()
            c_im=cartonize_image(img)
            st.image(c_im)
        click=st.sidebar.button('Generate Code')
        if(click):
            st.code('''
    def cartonize_image(our_image):
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img,1)
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        # Edges
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        #Color
        color = cv2.bilateralFilter(img, 9, 300, 300)
        #Cartoon
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        return cartoon
            ''')

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    Add_github_icon="""
    <a href='https://github.com/RheagalFire'><img height="32" width="32" src="https://cdn.jsdelivr.net/npm/simple-icons@v4/icons/github.svg" /></a> Follow me on Github
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.sidebar.markdown(Add_github_icon, unsafe_allow_html=True)
