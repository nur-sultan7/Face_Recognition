from unicodedata import name
import face_recognition
import os
from pymongo import MongoClient

#Объявление констант
#Путь к директории файлов
DIR_PATH = "files"

class DAO:
    known_face_encodings = []
    known_face_names = []
    known_paths = []
    
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
        known_paths=[]

        #Записывает в бд декодированные лица
        for i in self.faces.find():
            known_face_encodings.append(i["embedding"])
            known_face_names.append(i["name"])
            known_paths.append(i["imgpath"])
        return known_face_names, known_face_encodings,known_paths



    def insertIfNotExist(self, names, encodings,paths):
        for i in range(len(names)):
            self.faces.update_one({"name": names[i]}, {
                "$set":{"embedding": encodings[i].tolist()} 
                },  upsert= True)
            self.faces.update_one({"name": names[i]},{"$set":{"imgpath":paths[i]}}, upsert=True)

class Data:
        def loadDataFromDir(self):
            known_face_encodings = []
            known_face_names = []
            known_paths =[]
            
            # Load a sample picture and learn how to recognize it.
            for img in os.listdir(DIR_PATH):
                imgFile=face_recognition.load_image_file(os.path.join(DIR_PATH,img))
                known_paths.append(os.path.join(DIR_PATH,img))
                known_face_encodings.append(face_recognition.face_encodings(imgFile)[0])
                known_face_names.append(img.split(".")[0]) 
            return known_face_names, known_face_encodings, known_paths
