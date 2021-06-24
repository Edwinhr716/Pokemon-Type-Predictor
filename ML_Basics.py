import tensorflow as tf 
import pandas as pd 
import matplotlib.pyplot as plt
import tkinter as tk 
from ML_Model import *



allStats = []
predict = {}    
 

def getVal(iterator):
    Stat = allStats[iterator]
    return Stat



def input_fn(features, batch_size = 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

def prediction():
    features = ['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed']
    iterator = 0
    for feature in features:
        valid = True
        val = getVal(iterator)
        iterator += 1
        print(val)
        predict[feature] = [float(val)]

    for i in predict:
        print(i)

def output():        
    predictions = classifier.predict(input_fn = lambda: input_fn(predict))
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%)'.format(TYPES[class_id], 100 * probability))
    print('Prediction is "{}"'.format(TYPES[class_id]))
    output = 'Prediction is ' + TYPES[class_id]
    print(output)
    label['text'] = output
    

def printPredict():
    for i in predict:
        print(i)

def printAllStats():
    for i in allStats:
        print(i)


def getStats(hp, Atk, Def, Sp_A, Sp_D, Spd):
    hp = int(hp)
    Atk = int(Atk)
    Def = int(Def)
    Sp_A = int(Sp_A)
    Sp_D = int(Sp_D)
    Spd = int(Spd)
    allStats.append(hp)
    allStats.append(Atk)
    allStats.append(Def)
    allStats.append(Sp_A)
    allStats.append(Sp_D)
    allStats.append(Spd)
    printAllStats()
    #printPredict()
    prediction()
    output()
    allStats.clear()
    predict.clear()



root = tk.Tk()

canvas = tk.Canvas(root, height=700, width = 800)
canvas.pack()

frame_title = tk.Frame(root, bd = 5)
frame_title.place(relx = 0.5, rely = 0.05, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame_instructions = tk.Frame(root, bd = 5)
frame_instructions.place(relx = 0.5, rely = 0.1, relwidth = 0.75, relheight = 0.05, anchor = 'n')


frame1 = tk.Frame(root, bd = 5)
frame1.place(relx=0.5, rely = 0.2, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame2 = tk.Frame(root,  bd = 5)
frame2.place(relx=0.5, rely = 0.25, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame3 = tk.Frame(root, bd = 5)
frame3.place(relx=0.5, rely = 0.3, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame4 = tk.Frame(root,  bd = 5)
frame4.place(relx=0.5, rely = 0.35, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame5 = tk.Frame(root,  bd = 5)
frame5.place(relx=0.5, rely = 0.4, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame6 = tk.Frame(root,  bd = 5)
frame6.place(relx=0.5, rely = 0.45, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame7 = tk.Frame(root,  bd = 5)
frame7.place(relx=0.5, rely = 0.5, relwidth = 0.75, relheight = 0.05, anchor = 'n')

frame8 = tk.Frame(root, bd = 5)
frame8.place(relx = 0.3, rely = 0.9, relwidth = 0.75, relheight = 0.05, anchor = 'n')

HP = tk.Entry(frame1, font = 40)
HP.place(relwidth = 0.65, relheight = 1)

Attack = tk.Entry(frame2, font = 40)
Attack.place(relwidth = 0.65, relheight = 1)

Defense = tk.Entry(frame3, font = 40)
Defense.place(relwidth = 0.65, relheight = 1)

Sp_Atk = tk.Entry(frame4, font = 40)
Sp_Atk.place(relwidth = 0.65, relheight = 1)

Sp_Def = tk.Entry(frame5, font = 40)
Sp_Def.place(relwidth = 0.65, relheight = 1)

Speed = tk.Entry(frame6, font = 40)
Speed.place(relwidth = 0.65, relheight = 1)

button = tk.Button(frame8, text = "Predict", command = lambda: getStats(HP.get(), Attack.get(), Defense.get(), Sp_Atk.get(), Sp_Def.get(), Speed.get()))
button.place(relx = 0.5, relwidth = 0.45, relheight = 1)

label_HP = tk.Label(frame1, text = "HP", font = 40)
label_HP.place(relx = 0.7, relwidth = 0.45, relheight = 1)

label_Attack = tk.Label(frame2, text = "Attack", font = 40)
label_Attack.place(relx = 0.7, relwidth = 0.45, relheight = 1)

label_Defense = tk.Label(frame3, text = "Defense", font = 40)
label_Defense.place(relx = 0.7, relwidth = 0.45, relheight = 1)

label_Sp_Atk = tk.Label(frame4, text = "Sp. Atk", font = 40)
label_Sp_Atk.place(relx = 0.7, relwidth = 0.45, relheight = 1)

label_Sp_Def = tk.Label(frame5, text = "Sp. Def", font = 40)
label_Sp_Def.place(relx = 0.7, relwidth = 0.45, relheight = 1)

label_Speed = tk.Label(frame6, text = "Speed", font = 40)
label_Speed.place(relx = 0.7, relwidth = 0.45, relheight = 1)

label = tk.Label(frame7, font = 40)
label.place(relwidth = 1, relheight = 1)

label_title = tk.Label(frame_title, font = 40)
label_title.place(relwidth = 1, relheight = 1)

label_title['text'] = "Pokemon Type Predictor"

label_instructions = tk.Label(frame_instructions)
label_instructions.place(relwidth = 1, relheight = 1)


label_instructions['text'] = "Write the corresponding stats"

root.mainloop()



 



    