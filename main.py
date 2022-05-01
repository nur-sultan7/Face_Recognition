from cProfile import label
from email.mime import image
import face_recognition
import cv2
from tkinter import *
from PIL import ImageFont, ImageDraw, Image, ImageTk
import numpy as np
import os
from pymongo import MongoClient
from data_manager import Data, DAO
import asyncio



#Подключение к БД
# client = MongoClient('localhost', 27017)
# db = client['face_db']

# faces = db.face




name=[]
encodings=[]
dao = DAO()
data = Data()
names, encodings = data.loadDataFromDir()
dao.insertIfNotExist(names, encodings)


known_face_encodings = []
known_face_names = []

#Загружаем из бд декодированные лица и имена 

known_face_names, known_face_encodings = dao.getData()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

win =Tk()
win.geometry("800x600")
label=Label(win)
my_string_var = StringVar()
my_string_var.set("What should I learn")
name_label = Label(win, textvariable = my_string_var)
label.grid(row=0,column=0)
name_label.grid(row=1, column=0)


# Create arrays of known face encodings and their names
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []



def show_frame():
    process_this_frame = True
    # Grab a single frame of video
    _, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            #If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        b,g,r,a = 0,255,0,0

        # Draw a box around the face
        cv2.rectangle(frame, (left , top +10 ), (right, bottom), (0, 0, 255), 2)
        # черный  квадрат
        cv2.rectangle(frame, (left -150, bottom +35), (right +110, bottom), (0, 0, 0), cv2.FILLED)
        # Подпись
        cv2.putText(frame, name,  (left - 120, bottom +20 ), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        my_string_var.set(name)
       

    # # Display the resulting image
    # cv2.imshow('Video', frame)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk=imgtk
    label.configure(image=imgtk)
    label.after(20, show_frame)

    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     return
show_frame()
win.mainloop()


# Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
class FaceRecognitionWindow:
    def __init__(self):
        self.encodings=[]
        self.dao = DAO()
        self.data = Data()
        self.known_face_encodings = []
        self.known_face_names = []
        self.loaded_names=[]
        self.loaded_embendings=[]
        self.video_capture = cv2.VideoCapture(0)

    def loadDataFromDir(self):
        return self.data.loadDataFromDir()

    def insertDataFromDirIfNotExist(self, names, encodings):
        dao.insertIfNotExist(names, encodings)
    
    def loadDataFromDb(self):
        self.known_face_names, self.known_face_encodings = dao.getData()
    
    def CreateWindow(self):
        self.win =Tk()
        self.win.geometry("800x600")
        self.label=Label(win)
        self.label.grid(row=0,column=0)
