# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:26:57 2025

@author: marco
"""

# Comando incial
print("hello world 1!")
print("hello world 2!")
print("hello world 3!")
print("hello world 4!")


#  Variáveis
meuInteiro = 100
meuFloat = 3.1415
minhaString = "Hello world"
meuLogico = True
meuComplexo = 2+3j

# Listas
a = "muito"
b = "boa"
umaLista = ['uma', 'lista', a, b]
outraLista = [[4,5,6,7],[3,4,5,6]]
c = umaLista[1]
d = umaLista[-2]
e = umaLista[1:3]
f = umaLista[1:]
g = umaLista[:3]
h = umaLista[:]
i = outraLista[1][0]
j = outraLista[1][:2]

indice = umaLista.index(a)
conta = umaLista.count(a)
umaLista.append('!')
umaLista.remove('!')
del(meuLogico)
del(umaLista[0:1])
umaLista.reverse()
umaLista.pop(-2)
umaLista.insert(0,'!')
umaLista.sort()


#  Dicionários
pessoas = {
    "nome":"joão",
    "idade":18,
    "habilidades": ["python", "java", "php"]
}

pessoas["idade"] = 40

nome = pessoas["nome"]
idade = pessoas["idade"]
habilidades = pessoas["habilidades"]

#  Operações aritméticas
conta = (9 - (1 + 2)) / 3.0
divisaoInteira = 10 // 9
outraConta = 10**2 + conta

#  Operadores relacionais
teste = 1 > 2
teste2 = "a" > "b"
teste3 = 2**2 == 2*2

# Operadores logicos
x = True and False
y = True or False
z = not x

# entrada de dados
nome = input("Qual é o seu nome?")
idade = int(input("Quantos anos você tem?"))

# saída de dados
print(f"Seu nome é {nome} e você tem {idade} anos.")

#  Manipulação de Strings
string0 = "Marco"
string1 = string0*2
string2 = string0 + ' Kappel'
teste = 'pp' in string2
string3 = string1[1]
string4 = string2[-1]
string5 = string1[0:3]

string6 = string1.upper()
string7 = string1.lower()
conta = string2.count('p')
string8 = string2.replace('Kappel', 'Andre')
string9 = string2.strip()

#  Estrutras de decisão
i = 9
if i == 7:
    print("dentro do if")
    print("dentro do if")
elif i == 8:
    print("dentro do elif")
    print("dentro do elif")
else:
    print("dentro do else")
    print("dentro do else")
print("fora do if")

#  estruturas de repeticao
lista = [1,2,3,4,5]

# foreach
for item in lista:
    print(item)

# for
for num in range(0, 10, 2):
    print(num)

notas = {
    'Portugues': 7,
    'Matematica':9,
    'Logica':7,
    'Algoritmos':8    
}

for x, y in notas.items():
    print(x, ":", y)

contador = 0
while contador < 10:
    print("Valor do contador é:", contador)
    contador += 1

#  funções

def minhaFuncao(arg1, arg2):
    #  código da funcao
    return arg1

def soma(x, y, z=0):
    soma = x + y + z
    return soma

a = soma(1,3,5)

#  numpy arrays
import numpy as np

minhaLista = [1,2,3,4]
meuArray = np.array(minhaLista)
meuArray2D = np.array([[1,2,3],[4,5,6]])
indice = meuArray[1]
subArray = meuArray[0:2]
teste = meuArray > 3
arrayDobrado = meuArray * 2
somaVetorial = meuArray + arrayDobrado
dimensao = meuArray.shape

media = np.mean(meuArray)
desvio = np.std(meuArray)

#  Pandas
import pandas as pd

#  series
notas = pd.Series([2,7,5,10,6], index = ["João", "Maria", "Pedro", "Jennifer", "Luis"])
a = notas.values
b = notas.index

c = notas["Maria"]
media = notas.mean()
desvio = notas.std()

resumo = notas.describe()

#  Dataframe
dicionarioTurma = {'Aluno' :  ["João", "Maria", "Pedro", "Jennifer", "Luis"],
                   'Faltas': [3,4,2,1,4],
                   'Prova': [2,7,5,10,6],
                   'Seminário': [8.5,7.6,9.2,7.5,8.0]
}

df = pd.DataFrame(dicionarioTurma, index=['GSIS123','GSIS124','GSIS125','GSIS126','GSIS127'])
resumo = df.describe()

alunoEspecifico1 = df.loc['GSIS125']
alunoEspecifico2 = df.iloc[-1]

acima8 = df[df["Seminário"] > 8.0]






























