#------------------------------------------------------------------------------
#---- Segundo trabajo: Reconocimiento de caracteres escritos a mano usando ML --
#------------------------------------------------------------------------------
#---- Por: - Santiago Orozco Holguín ......... CC 1088318863 ------------------
#--------------- correo: santiago.orozcoh@udea.edu.co -------------------------
#------------------------------------------------------------------------------
#---------------- Estudiante Ingeniería Electrónica  -------------------------
#-------------------- Universidad de Antioquia --------------------------------
#----------- Curso: Procesamiento Digital de Imágenes II ----------------------
#---------------------- Fecha: Enero de 2022 ----------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#--0. Importar librerías ------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import cv2
import os

#from google.colab.patches import cv2_imshow

def extract_features_from_char(char_, n_features):
  n_lines = n_features  #16

  angle_div = int(360/n_lines)     #Angulo que coresponde para el numero de lineas
  kernel = np.ones((3,3))*255


  all_pts = []
  external_pts = []
  letter_features = []

  letter = char_      

  letter_inv = 255 - letter    

  contorno = np.ones(letter.shape)*255
  intersec_radial = np.ones(letter.shape)*255

  contours, hierarchy= cv2.findContours(letter_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

  cv2.drawContours(contorno, sorted_contours, -1, (0,255,0), 1) #Se dibuja el contorno
  #cv2.imshow("last",contorno)
  #------------------------------------------------------------------------------
  #-- 2. Centro de masa (momento) -----------------------------------------------
  #------------------------------------------------------------------------------

  cont_center = np.copy(contorno)   #Se crea copia de contorno

  M = cv2.moments(sorted_contours[0])      #Se encuentra el centro de masa

  if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  else:
    # set values as what you need in the situation
    print("HELPPPP")
    cX, cY = 0, 0

  cv2.circle(cont_center,(cX,cY),1,(0,0,0),-1)   #Se dibuja el centro de masa

  cont_radial = np.copy(cont_center)  #Se copia el contorno con el centro de masa

  #------------------------------------------------------------------------------
  #-- 3. Analisis por cada linea que sale desde el centro de masa ---------------
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
  #-- 4. De los todos los puntos encontrados se dejan solo los mas externos -----
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

  return norm_letter_features

def binarize(img, thresh):
  th, im_th = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

  return im_th

#------------------------------------------------------------------------------
#-- 1. Inicialización el sistema ----------------------------------------------
#------------------------------------------------------------------------------

n_lines = 24                     #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Numero de lineas que salen del centro de masa
angle_div = int(360/n_lines)     #Angulo que coresponde para el numero de lineas
kernel = np.ones((3,3))*255

number_list = [ "30", "31", "32", "33", "34", "35", "36", "37", "38", "39"]
uppercase_list = ["41", "42", "43", "44", "45", "46", "47", "48", "49", "4A", "4B", "4C", "4D", "4E", "4F", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5A"]
lowercase_list = ["61", "62", "63", "64", "65", "66", "67", "68", "69", "6A", "6B", "6C", "6D", "6E", "6F", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7A"]

df = pd.DataFrame(columns=range(0,n_lines))

char_list = number_list
#char_list = uppercase_list
#char_list = lowercase_list

print("\n", "*"*61, sep="")
print("* Extracting geometric features from characters in DataBase *")
print("*"*61)

for char_ in range(len(char_list)):   #len(char_list)
  print("\nCharacter being processed: 0x", char_list[char_],  sep="")
  char_folder_path = "./Temp/by_class/" + char_list[char_] + "/train_" + char_list[char_] 
  #print(char_folder_path)

  list = os.listdir(char_folder_path) # dir is your directory path
  number_files = len(list)
  #print number_files

  for n_char in range(1900): #range(number_files):   #how to count files in folder
    if n_char%100 == 0:
      print("\r\t >> ", n_char, " characters processed", end="")

    img_file = char_folder_path + "/train_" + char_list[char_] + "_" + "0"*(5-len(str(n_char)))+ str(n_char) + ".png"    #Se define el path de la imagen
    letter = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) #Se lee la imagen en escala de grises
    #letter_inv = 255 - letter 

    norm_letter_features = extract_features_from_char(letter, n_lines)
    #print("norm", len(norm_letter_features))
    temp_append = np.append(int(char_list[char_], 16), norm_letter_features)
    #print("XXXX",len(temp_append))
    #print("SHAPE:", df.shape)
    df.loc[len(df)] = temp_append

    #print(df)
  print("")

df.to_csv("number_features_24_1900.csv", index = False)     #Funcion para guardar el archivo csv
