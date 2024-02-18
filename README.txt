Librarii si versiuni:

Python 			  3.11.5
numpy                     1.26.2
opencv-python             4.8.1.78
pandas                    2.1.4
Pillow                    10.0.1
scikit-image              0.20.0
scikit-learn              1.3.2
tqdm                      4.66.1
ultralytics               8.0.231


Model folosit pentru bonus:
yolov8


Toate librariile si versiunile se gasesc si in fisierul environment.yaml.



Proiectul are la baza codul de la laborator, fiind modificate anumite functii. In afara codulului de laborator exista si doua fisiere ipynb, unul pentru a crea datele de antrenare si a testa diferite metode pentru task-ul 1, celalalt pentru task-ul 2. Alte trei fisiere sunt folosite pentru bonus, doua pentru setul de date si antrenare, iar main_final pentru predictii.

Task1 + Task2:
Fisier de rulat pentru task1 si task2: runFinal.py din directorul Solutie.
Pentru task 1 trebuie schimbat directorul pentru testare in Parameters (in loc de validare).

Fisier de rulat pentru bonus: main_final.ipynb din folderul yolo.
Pentru YOLO rezultate sunt scrise in folder-ul rezultate din acelasi director, trebuie doar inlocuite imaginile de validare cu cele pentru testare(celula 3): fisiere_det = ["validare/" + im_name for im_name in os.listdir("validare")]


