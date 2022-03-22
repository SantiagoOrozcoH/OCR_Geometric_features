---------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------- COMO CORRER EL PROGRAMA ---------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

ACLARACION: Este código fue realizado en un Macbook Air M1 (ARM) y no ha sido probado en otra arquitectura o sistema operativo

---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------
                 Se debe crear un ambiente virtual y posteriormente instalar los requerimientos de requirements.text
    
    python3 -m virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

               >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3_GUI_char_recognition.py      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

Este código es el código que resume todo el trabajo, permite abrir una imagen y digitalizar el texto
La imagen debe estar binarizada al momento de cargarla, las letras deben estar separadas unas de otras y las lineas deben estar rectas
		$ cd ./3_GUI/
                $ python3 3_GUI_char_recognition.py

Si se quiere utilizar otro modelo de los disponibles en la carpeta 3_GUI, es necesario sacarlo de la carpeta 2_ML y llevarlo a
la carpeta 3_GUI Se debe cambiar el nombre del archivo .csv que se lee desde el código y si se va a utilizar el modelo con 32 
características se debe cambiar la variable N_ML=24 a N_ML=32 

---------------------------------------------------------------------------------------------------------------------------------

1_Features_extraction.py

Este primer archivo se encarga de recorrer toda la base de datos y de extraer las características geométricas
de cada carácter. Se diferencian 3 grupos de caracteres
    - Mayusculas
    - Minúsculas
    - Numeros

Cambiando/descomentando la variable char_list se realiza el cambio de a que caracteres se deben extraer las características
La variable n_lines define cuantas características se van a extraer de todos los caracteres de la base de datos
Igualmente al final del código se debe cambiar el nombre del archivo csv

Este código depende de las imágenes que constituyen la base de datos mnist (

---------------------------------------------------------------------------------------------------------------------------------

2_ML.py

Este código realiza el entrenamiento de la red neuronal, necesita el archivo .csv correspondiente en la misma carpeta 
De necesitar cambiar de archivo, se debe cambiarle nombre en la variable que lleva el nombre del .csv
El código entrega un archivo .pkl el cual es el modelo de la red neuronal exportado para ser utilizado en otro codigo

archivos .csv:
    - lowercase_features_24_1900.csv
    - uppercase_features_24_1900.csv
    - numbers_features_24_1900.csv
    - lowercase_features_32_1900.csv

Se adjuntan 4 modelos entrenados: 
    - ML_24_LOW_model.pkl
    - ML_24_UP_model.pkl
    - ML_24_NUM_model.pkl
    - ML_32_LOW_model.pkl

---------------------------------------------------------------------------------------------------------------------------------
