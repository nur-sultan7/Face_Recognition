from cProfile import label
import tkinter
from turtle import bgcolor
import face_recognition
import cv2
from tkinter import *
from PIL import ImageFont, ImageDraw, Image, ImageTk
import numpy as np
from data_manager import Data, DAO
import imutils

#Объявление констант. 
UNKNOWN_PERSON_INDEX = -1
UNKNOWN_IMAGE_PATH = "img\\nobody.png"

name=[]
encodings=[]
paths = []
known_index = UNKNOWN_PERSON_INDEX
dao = DAO()
data = Data()
names, encodings, paths= data.loadDataFromDir()
dao.insertIfNotExist(names, encodings, paths)
known_face_encodings = []
known_face_names = []
known_paths=[]

#Загружаем из бд декодированные лица и имена 
known_face_names, known_face_encodings,known_paths = dao.getData()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

win =Tk()
win.geometry("1150x720+100+50")
win.configure(bg='black')
win.resizable(True, True)

left_frame = tkinter.Label(win, bg='red')
left_frame.grid( row=1,column=0, padx=10, pady =10, ipady=10)  

lbl_over_video = tkinter.Label (left_frame, text="Камера", bg = 'red',fg = "black",  font = ("Arial", 14) )
lbl_over_video.grid(ipady=10)
video=Label(left_frame)
video.grid(padx=10, pady=50)


btn1 = tkinter.Button(win, text = 'Загрузить нового человека', width=25)
btn1.grid(column=1, row=2, padx=50, pady=5)

#Строка (StringVar) c именем студента
stud_name_var = StringVar()
stud_name_var.set("Студент не обнаружен...")
#Label для отображения имени студента. Label реагирует на изменения в StringVar
#который ему передается в качестве параметра. То есть изменения StringVar(с именем студента) изменяет и Label


# Create arrays of known face encodings and their names
face_locations = []
face_encodings = []
face_names = []

# Чтобы создать один экземляр интерфейса, необходимо воспользоваться объектным подходом - созадть Класс "Программный интерфейс"
class ProgrammInterface:
    def __init__(self):
        self.right_frame = tkinter.Frame(win, bg='red')
        self.right_frame.grid( row=1,column=1, padx=10, pady = 10)
        self.lbl_over_photo = tkinter.Label (self.right_frame, text=" Фото из базы", bg = 'red',fg = "black",  font = ("Arial", 14) )
        self.lbl_over_photo.grid(ipady= 10)
        self.lblInputImage = Label(self.right_frame) 
# Создаем объект класса "ProgrammInterface". Это и будет объектом нашего интерфеса. Один на весь код
interface = ProgrammInterface()


##############################################################################

        # вызывает окно выбора фотографий и разсещает выбранную в окне
def show_known_student(known_index) :
        #Проверка входного индекса распознования. Если индекс -1, то это UNKNOWN_PERSON_INDEX
        img_path=""
        if known_index == UNKNOWN_PERSON_INDEX:
            img_path = UNKNOWN_IMAGE_PATH
        else:
            img_path = known_paths[known_index]


        #right_frame = tkinter.Frame(win, bg='red')
        #right_frame.grid( row=1,column=1, padx=10, pady = 10)  
        #lbl_over_photo = tkinter.Label (right_frame, text=" Фото из базы", bg = 'red',fg = "black",  font = ("Arial", 14) )
        #lbl_over_photo.grid(ipady= 10)
                        # Чтение входного изображения и изменение размера
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        # Для отображения входного изображения в графическом интерфейсе
        imageToShow = imutils.resize(image, height=400, width = 385)
                # Для отображение фотографии в цвете 
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow)
        img = ImageTk.PhotoImage(image = im)      
        #lblInputImage = Label(right_frame)  
        interface.lblInputImage.configure(image=img)
        interface.lblInputImage.Image = img
        
        name_label = Label(interface.right_frame, textvariable = stud_name_var, fg = "white", bg = "black", 
                                    text = "Имя Студента", justify="center", font = ("Arial", 14))
        interface.lblInputImage.grid( padx="20", pady="20",)
        name_label.grid(column=0, row = 2, pady = 5, ipady = 5, ipadx = 5)                                            



##########################################################################


def show_frame():
    process_this_frame = True
    global known_index
    # Grab a single frame of video
    _, frame = video_capture.read()

    # Делает картинку цветастой!!!
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    


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
            # Параметр tolerance регулирует точность сравнения чем ниже тем точнее, но ниже четырех
            # нужна картинка более высокого качества. По  умолчанию значение = 0,6. Самый оптимальный вариант который я нащупал это 0,5
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
           
            # До начала распознавания ставим студент не обнаружен и known_index - UNKNOWN_PERSON_INDEX (-1)
            name = "Студент не обнаружен"
            known_index = UNKNOWN_PERSON_INDEX
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                known_index = best_match_index

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
        cv2.rectangle(frame, (left-80 , top -100 ), (right-50, bottom-80), (0, 0, 255), 2)
        # черный  квадрат
        cv2.rectangle(frame, (left -80, bottom-80), (right -110, bottom-50), (0, 0, 0), cv2.FILLED)
        # Подпись
        cv2.putText(frame, name,  (left - 60, bottom-60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        
        #В StringVar меняется на новое имя(обнаруженное) 
        stud_name_var.set(name)
     

        show_known_student(known_index)
        
       

    # # Display the resulting image
    # cv2.imshow('Video', frame)``
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video.imgtk=imgtk
    video.configure(image=imgtk)
    video.after(20, show_frame)

    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     return

show_frame()
show_known_student(UNKNOWN_PERSON_INDEX)
win.mainloop()

cv2.destroyAllWindows()