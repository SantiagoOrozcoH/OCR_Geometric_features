#------------------------------------------------------------------------------
#--- Segundo trabajo: Reconocimiento de caracteres escritos a mano usando ML --
#------------------------------------------------------------------------------
#---- Por: - Santiago Orozco Holguín ......... CC 1088318863 ------------------
#--------------- correo: santiago.orozcoh@udea.edu.co -------------------------
#------------------------------------------------------------------------------
#---------------- Estudiante Ingeniería Electrónica  -------------------------
#-------------------- Universidad de Antioquia --------------------------------
#----------- Curso: Procesamiento Digital de Imágenes II ----------------------
#---------------------- Fecha: febrero de 2022 -------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#--0. Importar librerías ------------------------------------------------------
#------------------------------------------------------------------------------

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget,QFileDialog, QListWidget, QLabel, QGridLayout, QPushButton
from PyQt6.QtGui import QPixmap, QImage, qRgb
from PyQt6.QtCore import Qt
from pathlib import Path
import cv2
import numpy as np
import sys
import pickle
from scipy.signal import find_peaks, lfilter, peak_widths
from matplotlib import pyplot as plt


N_ML = 24

#------------------------------------------------------------------------------
#-- 2. Definicion de funciones -------------------------------------------------
#------------------------------------------------------------------------------
# load
with open('ML_24_NUM_model.pkl', 'rb') as f:
    print("ML MODEL LOADED")
    nn = pickle.load(f)


def binarize(img, thresh):                                          #Funcion para binarizar partiendo de un threshold
  th, im_th = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

  return im_th

def find_hist(img, ax=1, plot=False):                       #Funcion para encontrar el histograma de la imagen binarizada
  #img = 255-img                                            #1 horizontal, 0 vertical
  img_row_sum = np.sum(img,axis=ax).tolist()
  # if plot:
  #   plt.plot(img_row_sum)
  #   plt.show()

  return img_row_sum

def paragraph_finder(sum, tresh):                       #Funcion para encontrar las lineas de texto en la imagen
  #print(sum)
  parag = False
  paragraph_list = []
  upper = -1
  lower = -1

  for y, pix in enumerate(sum):                       #Hace uso de la variable parag que indica si se esta o no en una linea
    #print("y:",y, "pix:", pix)                       #Las lineas se definen en el histograma como el cambio de 0 hasta el retorno a 0
    #print(y)                                         #visto desde el historial de blancos horizontal
    if parag == False:
      if pix > tresh:
        parag = True
        upper = y - 2                               #parag en True indica que se encuentra en una linea
                                                    #parag en False indica que se encuentra fuera de la linea
    if parag == True:
      if pix <= tresh:
        parag = False
        lower = y + 2

    if (lower != -1) & (upper != -1):
        print("up:", upper, "\tlow:", lower)
        paragraph_list.append([upper,lower])
        upper = -1
        lower = -1
        
  print("Parag_list:", paragraph_list)
  #print("upper:", upper, "lower:", lower)
  return paragraph_list                         #Retorna los puntos en Y donde se encuentran las lineas

def letter_finder(sum, tresh):              #Funcion para encontrar las letras en una linea texto
  #print(sum)                               #Realiza la misma funcion de encontrar la lineas pero esta vez se realiza 
  parag = False                             #El histograma en vertical para extraer caracteres
  paragraph_list = []
  upper = -1
  lower = -1

  for y, pix in enumerate(sum):
    #print("y:",y, "pix:", pix)
    #print(y)
    if parag == False:
      if pix > tresh:
        parag = True                          #parag en True indica que se encuentra en una linea
                                              #parag en False indica que se encuentra fuera de la linea
        upper = y 
    
    if parag == True:
      if pix <= tresh:
        parag = False
        lower = y 

    if (lower != -1) & (upper != -1):
        #print("up:", upper, "\tlow:", lower)
        paragraph_list.append([upper,lower])
        upper = -1
        lower = -1
        
    
  #print("upper:", upper, "lower:", lower)
  return paragraph_list

def remove_not_paragraph(paragraph_list, thresh):             #Funcion para remover las secciones pequeñas por ruido 
  _paragraph_list = paragraph_list.copy()                     #que podrian caer en la lista de lineas
  for i in reversed(range(len(_paragraph_list))):
    #print(list_p[i])
    #print(list_p[i][1]-list_p[i][0])
    if (_paragraph_list[i][1]-_paragraph_list[i][0]) < thresh:
      _paragraph_list.remove(_paragraph_list[i])

  return _paragraph_list

def align_img(img):                                           #Funcion para alinear una imagen con cierto angulo
  img_parag = img.copy()
  nonZeroCoordinates = cv2.findNonZero(img_parag)

  imageCopy = img_parag.copy()

  for pt in nonZeroCoordinates:

      imageCopy = cv2.circle(imageCopy, (pt[0][0], pt[0][1]), 1, (255, 0, 0))

  box = cv2.minAreaRect(nonZeroCoordinates)
  boxPts = cv2.boxPoints(box)

  for i in range(4):

      pt1 = (boxPts[i][0], boxPts[i][1])
      pt2 = (boxPts[(i+1)%4][0], boxPts[(i+1)%4][1])
      cv2.line(imageCopy, pt1, pt2, (0,255,0), 2, cv2.LINE_AA)

  angle = box[2]
  if(angle < -45):
      angle = 90 + angle

  h, w = img_parag.shape

  scale = 1.
  center = (w/2., h/2.)
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = img_parag.copy()
  cv2.warpAffine(img_parag, M, (w, h), rotated, cv2.INTER_CUBIC, cv2.BORDER_REPLICATE )

  return rotated

def low_pass(sig, var):                                         #Funcion para suavizar la grafica del histograma
  n = var  # 11 the larger n is, the smoother curve will be
  b = [1.0 / n] * n
  a = 1

  filtered_hist = lfilter(b,a,sig) #LOWPASS FILTER
  return filtered_hist

def extract_lines_from_image(binary_img, graph=True):     #Funcion para extraer las lineas (img) de la imagen
                                                          #Hace uso de los puntos encontrados por paragraph_finder()
  line_hight = 20          # Cambiar, ajustar en calibracion     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   minimo alto de linea de texto (filtro)
  print("hist")
  sum = find_hist(binary_img, ax=1, plot=graph)
  max = np.max(sum)

  list_p = paragraph_finder(sum, 0)                   #1% del maximo -> cambiar, automatizar en calibracion
  print("parrafos:", len(list_p))
  list_p = remove_not_paragraph(list_p, line_hight)   #Elimina verdaderos negativos de parrafos (depende de un threshold de altura de linea)
  print("parrafos post remove:", len(list_p))

  return list_p



def extract_char_from_line(line):                   #Funcion para extraer los caracteres de la imagen

  hist_c = find_hist(line, ax=0, plot=False)
  max = np.max(hist_c)

  filtered_hist_c = low_pass(hist_c, 5)        #filtro pasa bajas para quitar altas frecuencias y dejar picos claros
  #plt.plot(filtered_hist_c, linewidth=2, linestyle="-", c="b") 

  indices_c = find_peaks(filtered_hist_c, threshold=0)[0]    #cambiar threshold    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  sep_m_c = peak_widths(filtered_hist_c, indices_c, rel_height=0.5)[0]

  list_p_c = letter_finder(hist_c, 0)        #Cambiar en calibracion 
                      #Se usa el paragraph finder porque es proceso similar a extraccion de caracteres
  #print("list:", list_p_c)
  #print("Characters per line:", len(list_p_c))

  str_line = []

  for i in range(len(list_p_c)):
    if list_p_c[i][0] > 2: 
      if i < len(list_p_c):
        #print("letter #", i)
        loc_start = list_p_c[i][0] - 6
        loc_end = list_p_c[i][1] + 6
        temp_char = line[:, loc_start:loc_end]
        temp = "char" + str(i)
        cv2.imshow(temp, temp_char)          #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< char
        #print("char #", i)
        norm_features = extract_features_from_char(temp_char, N_ML)           #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Features
        #print("MMM:", norm_features)
        if sum(norm_features) > 0:
          norm_features = norm_features.reshape(1, -1)
          temp_s = nn.predict(norm_features)            #Se realiza la prediccion de la letra desde las caracteristicas geometricas normalizadas<<
          #print("PREDICTED: ", chr(int(temp_s)))
        else:
          temp_s = 36

        str_line.append(chr(int(temp_s)))
        #temp_str = Char_Clasification(norm_features)    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #Como poner espacios???
  
  return str_line, list_p_c

def extract_features_from_char(char_, n_features):
  n_lines = n_features  

  angle_div = int(360/n_lines)     #Angulo que coresponde para el numero de lineas
  kernel = np.ones((3,3))*255


  all_pts = []
  external_pts = []
  letter_features = []

  letter = char_      

  letter_inv = 255 - letter    

  contorno = np.ones(letter.shape)*255
  intersec_radial = np.ones(letter.shape)*255

  contours, hierarchy= cv2.findContours(letter.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

  cv2.drawContours(contorno, sorted_contours, -1, (0,255,0), 1) #Se dibuja el contorno
  #cv2.imshow("last",contorno)
  #------------------------------------------------------------------------------
  #-- Centro de masa (momento) -----------------------------------------------
  #------------------------------------------------------------------------------

  cont_center = np.copy(contorno)   #Se crea copia de contorno

  M = cv2.moments(sorted_contours[0])      #Se encuentra el centro de masa

  if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  else:
    # set values as what you need in the situation
    print("Bad center")
    cX, cY = 0, 0

  cv2.circle(cont_center,(cX,cY),1,(0,0,0),-1)   #Se dibuja el centro de masa

  cont_radial = np.copy(cont_center)  #Se copia el contorno con el centro de masa

  #------------------------------------------------------------------------------
  #-- Analisis por cada linea que sale desde el centro de masa ---------------
  #------------------------------------------------------------------------------
  for i in range(n_lines):   
    if i > 0:                 
      feature_line = np.ones(letter.shape)*255
      line = np.ones(letter.shape)*255
      angle = i*angle_div         #Se define el angulo para cada ciclo for
      length = 200                #Se asigna un tamaño de linea12

      x2 = int(cX + length * np.cos(np.radians(angle)))  #Se encuentra X del 2do punto
      y2 = int(cY + length * np.sin(np.radians(angle)))  #Se encuentra Y del 2do punto 

      cv2.line(feature_line, (cX, cY), (x2, y2), (0,0,0), 1) #Se crea linea 
      cv2.line(cont_radial, (cX, cY), (x2, y2), (0,0,0), 1)  #Se añade linea a imagen total

      line = cv2.add(255 - cont_center, 255 - feature_line)  #Se suma la imagen del contorno y la linea

      letter_line = 255 - line         #Invierte la linea

      intersec_pts = (feature_line) + contorno #Imagen con puntos de interseccion con la linea

      pts = np.where(intersec_pts == 0)   #Se encuentra la posicion de estas intersecciones 

      all_pts.append(pts)   #Se agregan los puntos a una lista global
      intersec_radial = 255 - cv2.add(255-intersec_radial, 255-intersec_pts)  #Se agregan los puntos a una imagen total
  
  # print("all points")
  # cv2_imshow(intersec_radial)
  #------------------------------------------------------------------------------
  #-- De los todos los puntos encontrados se dejan solo los mas externos -----
  #------------------------------------------------------------------------------

  for i in all_pts:
    max_dist = 0
    x_temp = 0                  #Se inicializan en 0 las variables a usar
    y_temp = 0
    exter_pts = [[],[]]
    distance = 0

    for j in range(len(i[0])):
      y = i[0][j]
      x = i[1][j]

      distance = ((cX - x)**2 + (cY - y)**2)**0.5   #Se calcula la distancia de cada punto al centro de masa
 
      if distance > max_dist:     
        max_dist = distance       #Se queda con la mayor distancia 
        x_temp = x
        y_temp = y                #Se guarda el punto correspondiente a la mayor distancia.

    letter_features.append(max_dist)  #Se añade la distancia a la variable global de caracteristicas
    exter_pts[0].append(x_temp)       #Se añade x a la variable global de puntos externos
    exter_pts[1].append(y_temp)       #Se añade y a la variable global de puntos externos
    external_pts.append(exter_pts)    #Se añade x a la variable global de puntos externos

    external_radial = np.ones(letter.shape)*255

    for i in external_pts:
      y = i[0]
      x = i[1]
      external_radial[tuple((x,y))] = 0 

  # print("Puntos")
  # cv2_imshow(external_radial)

  max = np.max(letter_features)

  if max > 0:
    norm_letter_features = letter_features/max
    #print("Norm feat:",norm_letter_features)
  else:
    norm_letter_features = letter_features

  return norm_letter_features     #Se retorna el vector de caracteristicas normalizadas

def digitize(img_path, ):           #Funcion que integra las demas para digitalizar la imagen de texto en texto plano ascii

    binary_thresh = 140                    #Cambiar en calibracion     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    letter = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Se lee la imagen en escala de grises
    print("SHAPE: ", letter.shape)
    kernel = np.ones((3,3),np.uint8)
    
    letter_inv = 255-letter 

    binar_letter = binarize(letter_inv, binary_thresh)
    #cv2.imshow("letter",binar_letter)                      #<<<<<<<<<<<<<<<<<<<<<< binarized image
    lines = extract_lines_from_image(binar_letter, True)    #Extraccion de lineas de la imagen de texto
    #print(lines)
    print("Number of lines found:", len(lines))

    text = ""

    for num, line in enumerate(lines):        #Se itera sobre cada una de las lineas

        temp_img = binar_letter[line[0]:line[1]]
        stt = "line #" + str(num)
        if temp_img.shape[0] > 0:        
            
            img_line = binar_letter[lines[num][0]-6:lines[num][1]+6]

            # if num == len(lines)-2:         #<<<<<<<<<<<<<<<<<<<<<<
            #   cv2.imshow(stt,255 - img_line)   #Muestra todas las lineas extraidas en diferentes ventanas

            s, list_pt = extract_char_from_line(img_line)   #Se extrae cada char de cada linea de texto

            temp_min_char = 100000
            #print("LINE")
            for i in range(len(list_pt)):               #Se trata de buscar el espacio entre letras para encontrar el espacio de verdad
              temp = list_pt[i][1] - list_pt[i][0]
              #print("dif:", temp)
              if temp < temp_min_char:
                temp_min_char = temp

            for i in range(len(list_pt)-1):            #Se determina cuando hay salto de linea en el texto y los posibles espacios
              #print("pt", list_pt[i])
              temp = list_pt[i+1][0] - list_pt[i][1]
              if temp <= temp_min_char+1:                 #<<<<<<<<<<<<<<<<<<<<< calibrar para espacio
                text += s[i]
              else:
                text += s[i] + " "
              #print(temp)
            if len(s) != 0:
              text += s[-1] + "\n" 
            else:
              text += "\n"
     
    print("DONE:",text)


    return text

##-----------------------------------------------------------------------------------------------------------------------------
## Interfaz grafica
##-----------------------------------------------------------------------------------------------------------------------------

width = 1280
height = 720


class NotImplementedException:
    pass

gray_color_table = [qRgb(i, i, i) for i in range(256)]

def toQImage(im, copy=False):             #Funcion para convertir imagenes de numpy a imagenes de Qt
    if im is None:
        return QImage()

    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format.Format_RGB888)
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format.Format_ARGB32)
                return qim.copy() if copy else qim

    raise NotImplementedException


class Window(QMainWindow):        #Definicion de la ventana que sera la interfaz del programa
    im_path = ""
    def __init__(self):
        super().__init__()
 
        #self.acceptDrops()
        # set the title
        self.setWindowTitle("Handwritten recognition")          #Titulo de ventana
 
        # setting  the geometry of window
        self.setGeometry(10,10,width,height)                    #Tamaño de la ventana
 
        # creating label
        self.label = QLabel()
         
        # loading image
        self.pixmap = QPixmap(manuscript)                       #Se carga la imagen al objeto
 
        # adding image to label
        self.label.setPixmap(self.pixmap)
 
        # Optional, resize label to image size
        self.label.resize(640,720)

        #self.label.setAlignment(AlignCenter)
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        self.label2 = QLabel("")                              #Label para imagen
        self.label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label2.resize(100,50)

        self.label3 = QLabel("#3")                            #Label para texto
        self.label3.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.grid = QGridLayout()

        btn1 = QPushButton("Load Handwritten text")             #Texto de los botones y su creacion
        btn2 = QPushButton("Digitize text")                     #Texto de los botones y su creacion
        #btn3 = QPushButton("Word segmentation")

        btn1.setCheckable(True)
        btn1.clicked.connect(self.was_clicked_1)
        #btn1.clicked.connect(self.was_toggled)

        btn2.setCheckable(True)
        btn2.clicked.connect(self.was_clicked_2)
        #btn2.clicked.connect(self.was_toggled)


        self.grid.addWidget(btn1,1,0)                         #Ubicacion de los objetos sobre una malla columna fila
        self.grid.addWidget(btn2,1,1)
        self.grid.addWidget(self.label,0,0)
        self.grid.addWidget(self.label2, 0,1)
        #self.grid.addWidget(btn3, 0, 1)


        container = QWidget()
        container.setLayout(self.grid)

        self.setCentralWidget(container)

    def showDialog(self):

        home_dir = str(Path.home())
        fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)
        print(fname[0])

        return fname[0]

    def was_clicked_1(self):                  #Funcion para cuando se de click al primer boton (carga la imagen desde el PC)
        print("Clicked")
        self.im_path = self.showDialog()
        self.load_img(self.im_path)

    def was_clicked_2(self):                  #Funcion para click del segundo boton (digitaliza la imagen)
        print("#2,")
        if self.im_path != "":
            lines = digitize(self.im_path)
            self.label2.setText(lines)
            self.label2.setWordWrap(True)

    def was_toggled(self, checked):
        print("Checked?", checked)
        self.grid.addWidget(self.label3, 0,1)

    def load_img(self, img_path):
        resize_dim = (500, 640)

        temp_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        temp_img = cv2.resize(temp_img, resize_dim, interpolation = cv2.INTER_AREA)

        temp_img = toQImage(temp_img)
        self.pixmap = QPixmap(temp_img)
 
        # adding image to label
        self.label.setPixmap(self.pixmap)
    
# create pyqt5 app
App = QApplication(sys.argv)

imag = np.zeros((640,500))
imag = np.require(imag, np.uint8, 'C')
manuscript = toQImage(imag)                   #Se carga una imagen negra temporal 

# create the instance of our Window
window = Window()                           #Se crea el objeto ventana
window.show()                               #Se ejecuta la ventana
# start the app
sys.exit(App.exec())


