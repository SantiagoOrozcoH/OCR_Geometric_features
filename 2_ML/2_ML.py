#------------------------------------------------------------------------------
#---- Segundo trabajo: Reconocimiento de caracteres escritos a mano usando ML --
#------------------------------------------------------------------------------
#---- Por: - Santiago Orozco Holguín ......... CC 1088318863 ------------------
#--------------- correo: santiago.orozcoh@udea.edu.co -------------------------
#------------------------------------------------------------------------------
#---------------- Estudiantes Ingeniería Electrónica  -------------------------
#-------------------- Universidad de Antioquia --------------------------------
#----------- Curso: Procesamiento Digital de Imágenes II ----------------------
#---------------------- Fecha: Enero de 2022 ----------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#--0. Importar librerías ------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from matplotlib import pyplot as pyplot 
from sklearn.neural_network import MLPClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold, GridSearchCV

from sklearn.metrics import classification_report
from pretty_confusion_matrix import pp_matrix

#------------------------------------------------------------------------------
#--0. Definicion de funciones --------------------------------------------------
#------------------------------------------------------------------------------

def delete_useless_rows(data):
  rows = []
  for i in range(data.shape[0]):
    if all(elem == 0 for elem in data[i,:]):
      rows.append(i)
  return rows

def EvalRendMC(y_test_, y_pred_):

    CMatriz = confusion_matrix(y_test_, y_pred_)

    recall = recall_score(y_test_, y_pred_, average='macro')
    precision = precision_score(y_test_, y_pred_, average='macro')
    A = float(np.trace(CMatriz))/float(np.sum(CMatriz))*100

    return [A, precision, recall, CMatriz]


#df = pd.read_csv("lowercase_features_24_1900.csv") 
df = pd.read_csv("numbers_features_24_1900.csv")              #Se carga el archivo .csv con las caracteristicas de los caracteres
#frames = [df, df2]

#concatenate dataframes
#df_upper_low = pd.concat(frames, sort=False)

#df_upper_low.reset_index(drop=True, inplace=True)

y = df.iloc[:,0]                                              #Se extrae el vector de etiquetas
#y = df_upper_low.iloc[:,0]
print("y shape:",y.shape)
X = df.iloc[:, 1:]                                            #Se extraen las caracteristicas
#X = df_upper_low.iloc[:, 1:]
print("X shape:",X.shape)

del_rows = delete_useless_rows(X.values)                    #Se encuentras las filas que estan en ceros

#print(X.shape)
X = X.drop(del_rows, 0)         #Se eliminan las filas en ceros de las caracteristicas
#print(X.shape)
#print(y.shape)
y = y.drop(del_rows, 0)       #Se eliminan las filas en ceros de las etiquetas
#print(y.shape)

X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.2, random_state=42)   #Se dividen los datos entre entrenamiento y validacion

nn = MLPClassifier(activation="logistic", solver="adam", hidden_layer_sizes=(110,), random_state=42, max_iter = 300)
                                              #Se crea el clasificador de red neuronal con funcion de activacion rlogistic
                                              #Con 110 nueronas en la capa oculta
nn.fit(X_train, y_train)

y_pred = nn.predict(X_val)          #Se predicen las letras del test

wrong = []
good = []
n_bad = 0
n_good = 0

for i in range(len(y_pred)):
  if y_val[i] == y_pred[i]:
    good.append(i)
    n_good += 1
  
  else:
    wrong.append(i)
    n_bad += 1

print("Bad:", n_bad)
print("good:", n_good)

eval = EvalRendMC(y_pred, y_val)        #Se evalua rendimiento
print(eval[3])

df_cm = pd.DataFrame(eval[3], index=range(0, 10), columns=range(0, 10))

print('\n\nClassification Report\n')
print(classification_report(y_pred, y_val))     #Se imprime reporte de rendimiento

cmap = 'PuRd'
pp_matrix(df_cm, cmap=cmap)

# save
with open('ML_24_NUM_model.pkl','wb') as f:       #Se guarda modelo entrenado de red neuronal
    pickle.dump(nn,f)
