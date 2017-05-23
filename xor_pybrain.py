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
root.geometry("600x1030+400+100")
img = ImageTk.PhotoImage(Image.open('pybrain_logoe.gif'))
panel = tk.Label(root, image = img)
panel.grid(row=0, column=0, columnspan=2, rowspan=4, sticky=W+E+N+S, padx=5,
        pady=5)


#Set up misc. widgets
Label(root, text="Configurações", font=('Verdana','13','bold'), width=60,
    bg='#135823', fg='grey' ).grid(row=0,columnspan=2)
Label(root, text="Épocas", font=('Verdana','11','bold')).grid(row=1)
Label(root, text="Momento", font=('Verdana','11','bold')).grid(row=1,column=1)
Label(root, text="Aprendizagem", font=('Verdana','11','bold')).grid(row=3)

var1 = DoubleVar()
var2 = DoubleVar()
var3 = DoubleVar()
var4 = StringVar()
var5 = DoubleVar()
var6 = StringVar()

e1 = Scale( root, variable = var1, from_=0, to=10000, resolution=50,
        orient=HORIZONTAL)
e1.set(1000)

e2 = Scale( root, variable = var2, from_=0.0, to=1.0, resolution=0.01,
        orient=HORIZONTAL)
e2.set(0.90)

e3 = Scale( root, variable = var3, from_=0.001, to=0.9, resolution=0.001,
        orient=HORIZONTAL)
e3.set(0.01)

e1.grid(row=2)
e2.grid(row=2,column=1)
e3.grid(row=4)

# Activation Bias
Label(root, text="Ativar Bias", font=('Verdana','11','bold')).grid(row=3,
    column=0, columnspan=2,  sticky=W+E+N+S, padx=5, pady=5)
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

#Setting Weights
Label(root, text="Pesos", font=('Verdana','11','bold')).grid(row=3,
    column=1, columnspan=2)
e6 = StringVar()
e6.set("Padrão") # default value
var5 = OptionMenu(root, e6, 'Padrão', '(-1,0)', '(-1,1)', '(0,1)', '(-0.1,0.1)'
                ).grid(row=4, column=1, columnspan=2,)

# Run Button
submit = Button(root, text="Rodar", width=13, command=lambda: all(e1.get(),
            e2.get(),e3.get(),e4.get(),e5.get(),e6.get())).grid(row=7, column=0,
            pady=4,columnspan=2)

Label(root, text="Resultados", font=('Verdana','13','bold'), width=60,
    bg='#135823', fg='grey' ).grid(row=8,columnspan=2)

#Show total of epochs
Label(root,text='Numero de épocas percorridas: ',fg = 'red', font=('Verdana',
    '10',"bold")).grid(row=9,columnspan=2)
epo = IntVar()
epoc1 = Label(textvariable=epo, font=('Verdana','10',"bold")).grid(row=10,
            columnspan=2)

#Show error
Label(root, text='Erro final:',fg = 'red', font=('Verdana','10',"bold")).grid(
    row=11,columnspan=2)
er = StringVar()
Label(root,textvariable=er, font=('Verdana','10',"bold")).grid(row=12,
    columnspan=2)

#Show out 1
Label(root, text='\n0 XOR 0: Esperado = 0, Calculado:',fg = 'blue',
    font=('Verdana','10',"bold")).grid(row=13,columnspan=2)
m1 = DoubleVar()
Label(root,textvariable=m1, font=('Verdana','10',"bold")).grid(row=14,
    columnspan=2)

#Show out 2
Label(root, text='0 XOR 1: Esperado = 1, Calculado:',fg = 'blue',
    font=('Verdana','10',"bold")).grid(row=15,columnspan=2)
m2 = DoubleVar()
Label(root,textvariable=m2, font=('Verdana','10',"bold")).grid(row=16,
    columnspan=2)

#Show out 3
Label(root, text='1 XOR 0: Esperado = 1, Calculado:',fg = 'blue',
    font=('Verdana', '10',"bold")).grid(row=17,columnspan=2)
m3 = DoubleVar()
Label(root,textvariable=m3, font=('Verdana','10',"bold")).grid(row=18,
    columnspan=2)

#Show out 4
Label(root, text=' 1 XOR 1: Esperado = 0, Calculado:',fg = 'blue',
    font=('Verdana','10',"bold")).grid(row=19,columnspan=2)
m4 = DoubleVar()
Label(root,textvariable=m4, font=('Verdana','10',"bold")).grid(row=20,
    columnspan=2)


#Variables to make plots
it = DoubleVar()
err = DoubleVar()
sc = DoubleVar()
ds = DoubleVar()
dp = DoubleVar()

def all(e1, e2=0.0, e3=0.0, e4=True, e5="TanhLayer", e6='Padrão'):

    def rerun(epocas, e2, e3, e4, e5, e6):

        #Making the net
        #The first 3 parameters are the nember of layers: In-Hidden-Out
        net = buildNetwork(2, 4, 1, bias=e4, hiddenclass=eval(e5))

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

        #Creating training data
        global ds
        ds = SupervisedDataSet(2, 1)
        ds.addSample([1, 1], [0])
        ds.addSample([0, 0], [0])
        ds.addSample([0, 1], [1])
        ds.addSample([1, 0], [1])

        #Creating backdropTrainer
        trainer = BackpropTrainer(net, ds, learningrate=e3, momentum=e2)

        max_error = 1
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
            y_true = [0, 1, 1, 0]
            y_pred = [net.activate([0, 0])[0], net.activate([0, 1])[0],
                    net.activate([1, 0])[0], net.activate([1, 1])[0]]
            score = r2_score(y_true, y_pred)
            error = trainer.train()
            epocas = epocas - 1
            epocasPercorridas = epocasPercorridas + 1
            sc.append(score)
            err.append(error)
            it.append(epocasPercorridas)
            if error == 0:
                break

        #Show total of epochs
        global epo
        epo.set(epocasPercorridas)

        #Show error
        global er
        #er.set("%f "%(error))
        er.set(error)

        #Show out 1
        global m1
        m1.set("%f"%net.activate([1, 1])[0])
        m1.set(net.activate([1, 1])[0])

        #Show out 2
        global m2
        #m2.set("%f"%net.activate([1, 0])[0])
        m2.set(net.activate([1, 0])[0])

        #Show out 3
        global m3
        #m3.set("%f"%net.activate([0, 1])[0])
        m3.set(net.activate([0, 1])[0])

        #Show out 4
        global m4
        #m4.set("%f"%net.activate([0, 0])[0])
        m4.set(net.activate([0, 0])[0])

        global dp
        dp = np.array([net.activate([1, 1])[0], net.activate([1, 0])[0],
                    net.activate([0, 1])[0], net.activate([0, 0])[0]])

        root.update_idletasks()

        debug = True
        if debug:
            print '\n#########################################  DEBUG  ###########################################\n'

            print "\n\nPesos finais: ", net.params
            print "\nErro final: ", error
            print "\n\nTotal de epocas percorridas: ", epocasPercorridas
            print '\n\n1 XOR 1: Esperado = 0, Calculado = ', net.activate([1, 1])[0]
            print '1 XOR 0: Esperado = 1, Calculado =', net.activate([1, 0])[0]
            print '0 XOR 1: Esperado = 1, Calculado =', net.activate([0, 1])[0]
            print '0 XOR 0: Esperado = 0, Calculado =', net.activate([0, 0])[0]
            print 'Lista de erros', len(err)
            print 'Lista de it', len(it)
            print 'Lista de Scores', len(sc)

            print net['bias']
            print 'O DP é:', dp
            print "Pesos iniciais: ", p1
            print"shape", ps
            print"Novos pesos:", net.params
            print 'O tipo de e6: ', type(e6)
            print"Novo shape", net.params.shape
            print 'e6 =', e6
            print 'e5 =', e5

    rerun(e1, e2, e3, e4, e5, e6)

    root.mainloop()

#Create an empyt plotgrid
#Learning
fig1 = Figure(figsize=(5.5,5.15))
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.get_tk_widget().grid(row=22,column=0,columnspan=2)
canvas1.draw()

def plot_error ():
    fig1 = Figure(figsize=(5.5,5.15))
    a = fig1.add_subplot(111)
    a.plot(it, err,color='blue', linewidth=2)
    a.set_title('Curva de erro', fontsize=16)
    a.set_xlabel('Epoca', fontsize=14)
    a.set_ylabel('Erro', fontsize=14, labelpad=7)#.set_rotation(0)
    a.set_yscale('log')
    a.grid()

    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().grid(row=22,column=0,columnspan=2)
    canvas1.draw()

def plot_learn ():
    fig1 = Figure(figsize=(5.5,5.15))
    b = fig1.add_subplot(111)
    b.plot(it, sc,color='red', linewidth=2)
    b.set_title('Curva de Aprendizado', fontsize=16)
    b.grid()
    b.set_yscale('log')

    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().grid(row=22,column=0,columnspan=2)
    canvas1.draw()

def plot_points():#Need to be fixed. Is only a test for viewing
    fig1 = Figure(figsize=(5.5,5.15))
    a = fig1.add_subplot(111)
    b= fig1.add_subplot(111)
    a.plot(ds['input'],ds['target'],'bo',markersize=16, markeredgewidth=0)
    b.plot((-1),(-1),'bo',markersize=16, markeredgewidth=0, label='Esperado')
    a.plot(ds['input'],dp,'ro',markersize=10, markeredgewidth=0)
    b.plot((-1),(-1),'ro',markersize=10, markeredgewidth=0, label='Calculado')
    a.legend(loc='best', numpoints=1)
    a.grid()
    a.set_xlim([-0.2, 1.2])
    a.set_ylim([-0.2, 1.4])

    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().grid(row=22,column=0,columnspan=2)
    canvas1.draw()

#Plot error button
Button (root, text="Erro", command=plot_error).grid(row=21, column=0, pady=4)

#Plot learning button
Button (root, text="Aprendizado", command=plot_learn).grid(row=21, column=0,
    pady=4,columnspan=2)

#Plot points button
Button (root, text="Pontos", command=plot_points).grid(row=21, column=1, pady=4)

Label(root, text=u"\u00a9 Collumbus.2017", font=('Verdana','9'),
    foreground="#5c5555").grid(row=23,columnspan=2)

mainloop()
