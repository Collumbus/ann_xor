#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Collumbus'

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SigmoidLayer, LinearLayer
from pybrain.datasets import SupervisedDataSet
from sklearn.metrics import r2_score
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import Tkinter as tk
from Tkinter import *
from PIL import ImageTk, Image
import os

matplotlib.use('TkAgg')

#Window root configs
root = tk.Tk()
root.title("RNA para previsão de valores da porta XOR")
root.geometry("1260x1030+400+100")

#### Just to work on Mac
#img = ImageTk.PhotoImage(Image.open('pybrain_logoe.gif'))
#panel = tk.Label(root, image = img)
#panel.grid(row=0, column=0, columnspan=2, rowspan=4, sticky=W+E+N+S, padx=5,
#        pady=5)

img1 = ImageTk.PhotoImage(Image.open('pybrain_logoe.gif'))
panel1 = tk.Label(root, image = img1)
panel1.grid(row=0, column=0, columnspan=2, rowspan=4, sticky=W+E+N+S, padx=5,
        pady=5)


#Set up misc. widgets
# Settings Label
Label(root, text="Configurações", font=('Verdana','13','bold'), width=60,
    bg='#135823', fg='grey' ).grid(row=0,columnspan=2)

# Epochs
Label(root, text="Épocas", font=('Verdana','11','bold')).grid(row=1)
var1 = IntVar()
e1 = Entry( root, text = var1)
e1.grid(row=2)
var1.set(1000)

# Max Error
Label(root, text="Erro Máximo", font=('Verdana','11','bold')).grid(row=3)
var6 = DoubleVar()
e7 = Entry( root, text = var6)
var6.set(0.0001)
e7.grid(row=4)

# Momentum
Label(root, text="Momento", font=('Verdana','11','bold')).grid(row=1,column=1)
var2 = DoubleVar()
e2 = Scale( root, variable = var2, from_=0.0, to=1.0, resolution=0.01,
        orient=HORIZONTAL)
e2.grid(row=2,column=1)
e2.set(0.90)

# Learning Rate
Label(root, text="Aprendizagem", font=('Verdana','11','bold')).grid(row=3,
    column=1)
var3 = DoubleVar()
e3 = Scale( root, variable = var3, from_=0.001, to=0.9, resolution=0.001,
        orient=HORIZONTAL)
e3.grid(row=4, column=1)
e3.set(0.01)

# Activation Bias
Label(root, text="Ativar Bias", font=('Verdana','11','bold')).grid(row=3,
    column=0, columnspan=2)
e4 = IntVar(value=True)
chk = Checkbutton(root, variable=e4,onvalue=True, offvalue=False).grid(row=4,
                column=0, columnspan=2)

# Activation Function
Label(root, text="Função de Ativação", font=('Verdana','11','bold')).grid(row=5,
    column=0, columnspan=2)
e5 = StringVar()
e5.set("TanhLayer") # default value
var4 = OptionMenu(root, e5, 'TanhLayer', 'SigmoidLayer').grid(row=6, column=0,
                columnspan=2,)
var4 = StringVar()

#Setting Weights
Label(root, text="Pesos", font=('Verdana','11','bold')).grid(row=5,
    column=0)
e6 = StringVar()
e6.set("Padrão") # default value
var5 = OptionMenu(root, e6, 'Padrão', '(-1,0)', '(-1,1)', '(0,1)', '(-0.1,0.1)'
                ).grid(row=6, column=0)
var5 = DoubleVar()

# Questions Label
Label(root, text="Questionário", font=('Verdana','13','bold'), width=60,
    bg='#135823', fg='grey' ).grid(row=0, column=3, columnspan=2)

#Label(root, height=20, width=2, bg='#135823', fg='grey' ).grid(row=0, column=1,rowspan=9, sticky=E)
#Label(root, height=20, width=2, bg='#135823', fg='grey' ).grid(row=0, column=0,rowspan=9, sticky=W)

# Question 1
Label(root, text="1) É uma cirurgia de emergência?", font=('Verdana','11','bold')).grid(row=1,
    column=3,sticky=W)
q1 = StringVar()
q1.set("Não") # default value
qvar1 = OptionMenu(root, q1, 'Sim', 'Não').grid(row=2, column=3,sticky=W)
qvar1 = StringVar()

# Question 2
Label(root, text="2) Há condições cardíacas ativas?", font=('Verdana','11','bold')).grid(row=3,
    column=3,sticky=W)
q2 = StringVar()
q2.set("Não") # default value
qvar2 = OptionMenu(root, q2, 'Sim', 'Não').grid(row=4, column=3,sticky=W)
qvar2 = StringVar()

# Question 3
Label(root, text="3) O risco cirúrgico é baixo?", font=('Verdana','11','bold')).grid(row=5,
    column=3,sticky=W)
q3 = StringVar()
q3.set("Não") # default value
qvar3 = OptionMenu(root, q3, 'Sim', 'Não').grid(row=6, column=3,sticky=W)
qvar3 = StringVar()

img2 = ImageTk.PhotoImage(Image.open('risc.gif'))
panel2 = tk.Label(root, image = img2)
panel2.grid(row=7, column=3, columnspan=3, rowspan=8, sticky=W+E+N+S, padx=5,
        pady=5)

# Question 4
Label(root, text="4) A capacidade funcional é maior ou igual a 4 MET's e sem sintomas?",
        font=('Verdana','11','bold')).grid(row=15, column=3,sticky=W)
q4 = StringVar()
q4.set("Não") # default value
qvar4 = OptionMenu(root, q4, 'Sim', 'Não').grid(row=16, column=3,sticky=W)
qvar4 = StringVar()

# Question 5
Label(root, text="5)Existem quantos fatores clínicos de risco?",
        font=('Verdana','11','bold')).grid(row=17, column=3,sticky=W)
q5 = StringVar()
q5.set("Nenhum") # default value
qvar5 = OptionMenu(root, q5, 'Nenhum', '1 ou 2', '3 ou mais').grid(row=18, column=3,sticky=W)
qvar5 = StringVar()

img3 = ImageTk.PhotoImage(Image.open('risc2.gif'))
panel3 = tk.Label(root, image = img3)
panel3.grid(row=19, column=3, columnspan=3, rowspan=6, sticky=W+E+N, padx=5,
        pady=5)

def runi():
    print net.activate([0, 0, 0, 0, 0])[0]

rrt = IntVar(value=True)
rrf = IntVar(value=False)

# Train Button
submit = Button(root, bg='#98FB98', activebackground='#FF7F50', text="Treinar", width=13, command=lambda: all(int(e1.get()),
            e2.get(),e3.get(),e4.get(),e5.get(),e6.get(),float(e7.get()),
            q1.get(),q2.get(),q3.get(),q4.get(),q5.get(),rrt.get())).grid(row=7, column=0,
            pady=4)
#Run button
Button (root, text="Rodar", bg='#98FB98', activebackground='#FF7F50', command=lambda: all(int(e1.get()),
            e2.get(),e3.get(),e4.get(),e5.get(),e6.get(),float(e7.get()),
            q1.get(),q2.get(),q3.get(),q4.get(),q5.get(), rrf.get())).grid(row=7, column=1,
            pady=4)

# Results Label
Label(root, text="Resultados", font=('Verdana','13','bold'), width=60,
    bg='#135823', fg='grey' ).grid(row=8,columnspan=2)

#Show total of epochs
Label(root,text='Numero de épocas percorridas: ',fg = 'red', font=('Verdana',
    '11',"bold")).grid(row=9,columnspan=2)
epo = IntVar()
epoc1 = Label(textvariable=epo, font=('Verdana','11',"bold")).grid(row=10,
            columnspan=2)

#Show error
Label(root, text='Erro final:',fg = 'red', font=('Verdana','11',"bold")).grid(
    row=11,columnspan=2)
er = StringVar()
Label(root,textvariable=er, font=('Verdana','11',"bold")).grid(row=12,
    columnspan=2)

#Show out esp
Label(root, text='\nA classificação indicada ao paciente é:',fg = 'blue',
    font=('Verdana','11',"bold")).grid(row=13,columnspan=2)
m1 = StringVar()
Label(root, textvariable=m1, font=('Verdana','13',"bold")).grid(row=14,
    columnspan=2)

#Variables to make plots
it = DoubleVar()
err = DoubleVar()
sc = DoubleVar()
ds = DoubleVar()
dp = DoubleVar()


def all(e1, e2=0.0, e3=0.0, e4=True, e5="TanhLayer", e6='Padrão', e7=0.0001, q1='Não', q2='Não', q3='Não', q4='Não', q5='Não', rr=True):

    def rerun(epocas, e2, e3, e4, e5, e6, e7, q1, q2, q3, q4, q5):

        #Making the net
        #The first 3 parameters are the nember of layers: In-Hidden-Out
        global net
        net = buildNetwork(5, 4, 1, bias=e4, hiddenclass=eval(e5))

        p1 = net.params
        ps = net.params.shape

        #Setting Weights
        if e6 == '(-1,0)':
            net._setParameters(np.random.uniform(-1.0,0.0,net.params.shape[0]))
        elif e6 == '(-1,1)':
            net._setParameters(np.random.uniform(-1.0,1.0,net.params.shape[0]))
        elif e6 == '(0,1)':
            net._setParameters(np.random.uniform(0.0,1.0,net.params.shape[0]))
        elif e6 == '(-0.1,0.1)':
            net._setParameters(np.random.uniform(-0.1,0.1,net.params.shape[0]))

        ################# Instantiating the weights correctly to show ##########
        w_instance = []
        w_instance = net.params.tolist()

        #Creating training data
        global ds
        ds = SupervisedDataSet(5, 1)
        ds.addSample([0, 0, 0, 0, 0], [0])
        ds.addSample([0, 0, 0, 0, 1], [1])
        ds.addSample([0, 0, 0, 0, 2], [2])
        ds.addSample([0, 0, 0, 1, 0], [2])
        ds.addSample([0, 0, 0, 1, 1], [0])
        ds.addSample([0, 0, 0, 1, 2], [0])
        ds.addSample([0, 0, 1, 0, 0], [0])
        ds.addSample([0, 0, 1, 0, 1], [0])
        ds.addSample([0, 0, 1, 0, 2], [0])
        ds.addSample([0, 0, 1, 1, 0], [0])
        ds.addSample([0, 0, 1, 1, 1], [0])
        ds.addSample([0, 0, 1, 1, 2], [0])
        ds.addSample([0, 1, 0, 0, 0], [0])
        ds.addSample([0, 1, 0, 0, 1], [0])
        ds.addSample([0, 1, 0, 0, 2], [0])
        ds.addSample([0, 1, 0, 1, 0], [0])
        ds.addSample([0, 1, 0, 1, 1], [0])
        ds.addSample([0, 1, 0, 1, 2], [0])
        ds.addSample([0, 1, 1, 0, 0], [0])
        ds.addSample([0, 1, 1, 0, 1], [0])
        ds.addSample([0, 1, 1, 0, 2], [0])
        ds.addSample([0, 1, 1, 1, 0], [0])
        ds.addSample([0, 1, 1, 1, 1], [0])
        ds.addSample([0, 1, 1, 1, 2], [0])
        ds.addSample([1, 0, 0, 0, 0], [0])
        ds.addSample([1, 0, 0, 0, 1], [0])
        ds.addSample([1, 0, 0, 0, 2], [0])
        ds.addSample([1, 0, 0, 1, 0], [0])
        ds.addSample([1, 0, 0, 1, 1], [0])
        ds.addSample([1, 0, 0, 1, 2], [0])
        ds.addSample([1, 0, 1, 0, 0], [0])
        ds.addSample([1, 0, 1, 0, 1], [0])
        ds.addSample([1, 0, 1, 0, 2], [0])
        ds.addSample([1, 0, 1, 1, 0], [0])
        ds.addSample([1, 0, 1, 1, 1], [0])
        ds.addSample([1, 0, 1, 1, 2], [0])
        ds.addSample([1, 1, 0, 0, 0], [0])
        ds.addSample([1, 1, 0, 0, 1], [0])
        ds.addSample([1, 1, 0, 0, 2], [0])
        ds.addSample([1, 1, 0, 1, 0], [0])
        ds.addSample([1, 1, 0, 1, 1], [0])
        ds.addSample([1, 1, 0, 1, 2], [0])
        ds.addSample([1, 1, 1, 0, 0], [0])
        ds.addSample([1, 1, 1, 0, 1], [0])
        ds.addSample([1, 1, 1, 0, 2], [0])
        ds.addSample([1, 1, 1, 1, 0], [0])
        ds.addSample([1, 1, 1, 1, 1], [0])

        #Creating backdropTrainer
        trainer = BackpropTrainer(net, ds, learningrate=e3, momentum=e2)

        #max_error = 1
        error = 0.00001
        epocasPercorridas = 0

        #Training compared by error or epochs
        global it
        global err
        global sc
        err = []
        it = []
        sc = []
        score = 0
        while epocas > 0:
            y_true = [0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0]
            y_pred = [net.activate([0, 0, 0, 0, 0])[0],
                        net.activate([0, 0, 0, 0, 1])[0],
                        net.activate([0, 0, 0, 0, 2])[0],
                        net.activate([0, 0, 0, 1, 0])[0],
                        net.activate([0, 0, 0, 1, 1])[0],
                        net.activate([0, 0, 0, 1, 2])[0],
                        net.activate([0, 0, 1, 0, 0])[0],
                        net.activate([0, 0, 1, 0, 1])[0],
                        net.activate([0, 0, 1, 0, 2])[0],
                        net.activate([0, 0, 1, 1, 0])[0],
                        net.activate([0, 0, 1, 1, 1])[0],
                        net.activate([0, 0, 1, 1, 2])[0],
                        net.activate([0, 1, 0, 0, 0])[0],
                        net.activate([0, 1, 0, 0, 1])[0],
                        net.activate([0, 1, 0, 0, 2])[0],
                        net.activate([0, 1, 0, 1, 0])[0],
                        net.activate([0, 1, 0, 1, 1])[0],
                        net.activate([0, 1, 0, 1, 2])[0],
                        net.activate([0, 1, 1, 0, 0])[0],
                        net.activate([0, 1, 1, 0, 1])[0],
                        net.activate([0, 1, 1, 0, 2])[0],
                        net.activate([0, 1, 1, 1, 0])[0],
                        net.activate([0, 1, 1, 1, 1])[0],
                        net.activate([0, 1, 1, 1, 2])[0],
                        net.activate([1, 0, 0, 0, 0])[0],
                        net.activate([1, 0, 0, 0, 1])[0],
                        net.activate([1, 0, 0, 0, 2])[0],
                        net.activate([1, 0, 0, 1, 0])[0],
                        net.activate([1, 0, 0, 1, 1])[0],
                        net.activate([1, 0, 0, 1, 2])[0],
                        net.activate([1, 0, 1, 0, 0])[0],
                        net.activate([1, 0, 1, 0, 1])[0],
                        net.activate([1, 0, 1, 0, 2])[0],
                        net.activate([1, 0, 1, 1, 0])[0],
                        net.activate([1, 0, 1, 1, 1])[0],
                        net.activate([1, 0, 1, 1, 2])[0],
                        net.activate([1, 1, 0, 0, 0])[0],
                        net.activate([1, 1, 0, 0, 1])[0],
                        net.activate([1, 1, 0, 0, 2])[0],
                        net.activate([1, 1, 0, 1, 0])[0],
                        net.activate([1, 1, 0, 1, 1])[0],
                        net.activate([1, 1, 0, 1, 2])[0],
                        net.activate([1, 1, 1, 0, 0])[0],
                        net.activate([1, 1, 1, 0, 1])[0],
                        net.activate([1, 1, 1, 0, 2])[0],
                        net.activate([1, 1, 1, 1, 0])[0],
                        net.activate([1, 1, 1, 1, 1])[0]]

            score = r2_score(y_true, y_pred)
            error = trainer.train()
            epocas = epocas - 1
            epocasPercorridas = epocasPercorridas + 1
            sc.append(score)
            err.append(error)
            it.append(epocasPercorridas)
            if error < e7:
                break

        #Show total of epochs
        global epo
        epo.set(epocasPercorridas)

        #Show error
        global er
        #er.set("%f "%(error))
        er.set(error)

        #Specialist input
        esp = np.array([q1,q2,q3,q4])
        esp = np.where(esp == 'Sim', 1,0)
        if q5 == 'Nenhum':
            esp =np.append(esp,0)
        elif q5 == '1 ou 2':
            esp = np.append(esp,1)
        elif q5 == '3 ou mais':
            esp = np.append(esp,2)
        global pred_esp
        pred_esp = net.activate(esp)[0]

        #Show out esp
        global m1
        if -0.2 < pred_esp < 0.2:
            m1.set('Classe I')
        elif 0.8 < pred_esp < 1.2:
            m1.set('Classe IIb')
        elif 1.8 < pred_esp < 2.2:
            m1.set('Classe IIa')

        global dp
        dp = np.array([net.activate([0, 0, 0, 0, 0])[0],
                    net.activate([0, 0, 0, 0, 1])[0],
                    net.activate([0, 0, 0, 0, 2])[0],
                    net.activate([0, 0, 0, 1, 0])[0],
                    net.activate([0, 0, 1, 0, 0])[0],
                    net.activate([0, 1, 0, 0, 0])[0],
                    net.activate([1, 0, 0, 0, 0])[0]])

        root.update_idletasks()

        debug = True
        if debug:
            print '\n#########################################  DEBUG  ###########################################\n'

            print "\n\nPesos finais: ", net.params
            print "\nErro final: ", error
            print "\n\nTotal de epocas percorridas: ", epocasPercorridas
            print '\n\nSIARP_net 0: Esperado = 0, Calculado = ', net.activate([0, 0, 0, 0, 0])[0]
            print 'SIARP_net 1: Esperado = 1, Calculado =', net.activate([0, 0, 0, 0, 1])[0]
            print 'SIARP_net 2: Esperado = 2, Calculado =', net.activate([0, 0, 0, 0, 2])[0]
            print 'SIARP_net 3: Esperado = 2, Calculado =', net.activate([0, 0, 0, 1, 0])[0]
            print 'SIARP_net 4: Esperado = 0, Calculado =', net.activate([0, 0, 1, 0, 0])[0]
            print 'SIARP_net 5: Esperado = 0, Calculado =', net.activate([0, 1, 0, 0, 0])[0]
            print 'SIARP_net 6: Esperado = 0, Calculado =', net.activate([1, 0, 0, 0, 0])[0]

            print net['bias']
            print 'O DP é:', dp
            print "Pesos iniciais: ", p1
            print"Novos pesos:", np.array(w_instance)
            print"Score:", score
            print 'e6 =', e6
            print 'e5 =', e5
            print 'pred_esp:', pred_esp

    if rr:
        rerun(e1, e2, e3, e4, e5, e6, e7, q1, q2, q3, q4, q5)

    #Specialist input
    esp = np.array([q1,q2,q3,q4])
    esp = np.where(esp == 'Sim', 1,0)
    if q5 == 'Nenhum':
        esp =np.append(esp,0)
    elif q5 == '1 ou 2':
        esp = np.append(esp,1)
    elif q5 == '3 ou mais':
        esp = np.append(esp,2)
    global pred_esp
    pred_esp = net.activate(esp)[0]

    #Show out esp
    global m1
    if -0.2 < pred_esp < 0.2:
        m1.set('\nClasse I:\nBenefício >>>risco, cirurgia indicada')
    elif 0.8 < pred_esp < 1.2:
        m1.set('\nClasse IIb:\nBenefício >>risco, cirurgia provavelmente indicada')
    elif 1.8 < pred_esp < 2.2:
        m1.set('\nClasse IIa:\nBenefício > ou igual, risco indicação cirúrgica \npode ser considerada')
    print 'pred_esp:', pred_esp

    root.mainloop()

#Create an empyt plotgrid
#Learning
fig1 = Figure(figsize=(6.3,5.15))
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.get_tk_widget().grid(row=20,column=0,columnspan=2)
canvas1.draw()

def plot_error ():
    fig1 = Figure(figsize=(6.3,5.15))
    a = fig1.add_subplot(111)
    a.plot(it, err,color='blue', linewidth=2)
    a.set_title('Curva de erro', fontsize=16)
    a.set_xlabel('Epoca', fontsize=14)
    a.set_ylabel('Erro', fontsize=14, labelpad=7)#.set_rotation(0)
    a.set_yscale('log')
    a.grid()

    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().grid(row=20,column=0,columnspan=2)
    canvas1.draw()

def plot_learn ():
    fig1 = Figure(figsize=(6.3,5.15))
    b = fig1.add_subplot(111)
    b.plot(it, sc,color='red', linewidth=2)
    b.set_title('Curva de Aprendizado', fontsize=16)
    b.grid()
    b.set_yscale('log')

    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().grid(row=20,column=0,columnspan=2)
    canvas1.draw()

#Plot error button
Button (root, text="Erro", command=plot_error).grid(row=18, column=0, pady=4)

#Plot learning button
Button (root, text="Aprendizado", command=plot_learn).grid(row=18, column=1,
    pady=4)

Label(root, text=u"\u00a9 Collumbus.2017", font=('Verdana','9'),
    foreground="#5c5555").grid(row=21,columnspan=2)

mainloop()
