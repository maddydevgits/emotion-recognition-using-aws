import cv2
import boto3
import time
import streamlit as st

ACCESS_ID=''
ACCESS_KEY=''
REGION='us-east-1'
confi=0
status=''
flag=0

st.title('Emotion Recognition using AWS')
run = st.checkbox('Run Camera')
FRAME_WINDOW = st.image([])
#FRAME_WINDOW1=st.text_area()
    
camera = cv2.VideoCapture(0)
if st.button('recognize'):
    flag=1
while run:
        _, frame = camera.read()
        dummy=frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        if flag==1:
            flag=0
            cv2.imwrite('image.png',frame)
            client=boto3.client('rekognition',region_name=REGION,aws_access_key_id=ACCESS_ID,aws_secret_access_key= ACCESS_KEY)
            imageSource=open('image.png','rb')
            response = client.detect_faces(
            Image={'Bytes': imageSource.read()},Attributes=['ALL'])
            # st.write(response)
            for faceDetail in response['FaceDetails']:
                for emotion in faceDetail['Emotions']:
                    # st.write (emotion)
                    dummy=emotion['Confidence']
                    if(dummy>confi):
                        confi=dummy
                        status=emotion['Type']
            st.write (confi, status)

camera.release()

