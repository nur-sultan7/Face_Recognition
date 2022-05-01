from unicodedata import name
import face_recognition
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os
from pymongo import MongoClient

class DAO:
    known_face_encodings = []
    known_face_names = []
    def __init__(self):
        self.faces= self.connectDB()
        

    def connectDB(self):
        #Подключение к БД
            client = MongoClient('localhost', 27017)
            db = client['face_db']
            return db.face

    def getData(self):
        known_face_encodings = []
        known_face_names = []

        #Записывает в бд декодированные лица
        for i in self.faces.find():
            known_face_encodings.append(i["embedding"])
            known_face_names.append(i["name"])
        return known_face_names, known_face_encodings



    def insertIfNotExist(self, names, encodings):
        for i in range(len(names)):
            self.faces.update_one({"name": names[i]}, {"$set":{"embedding": encodings[i].tolist()} },  upsert= True)

class Data:
        def loadDataFromDir(self):
            known_face_encodings = []
            known_face_names = []
            # Load a sample picture and learn how to recognize it.
            for img in os.listdir("C:\\Users\\Nursultan\\Desktop\\recognition test\\files"):
                imgFile=face_recognition.load_image_file(os.path.join("C:\\Users\\Nursultan\\Desktop\\recognition test\\files",img))
                known_face_encodings.append(face_recognition.face_encodings(imgFile)[0])
                nameImg = img.split(".")
                known_face_names.append(nameImg[0]) 
            return known_face_names, known_face_encodings
