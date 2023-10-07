

import numpy as np

import pandas as pd
import streamlit as st 
import json
import string



tagset={'ADJ': 0, 'ADP': 1, 'ADV': 2, 'CONJ': 3, 'DET': 4, 'NOUN': 5, 'NUM': 6, 'PRON': 7, 'PRT': 8, 'VERB': 9, 'X': 10, '.': 11}
T = np.load('Transition_Prob.npy',allow_pickle=True)
E = np.load('Emission_Prob.npy',allow_pickle=True)
tagkeys = ['ADJ' ,'ADP' ,'ADV', 'CONJ', 'DET', 'NOUN', 'NUM' ,'PRON', 'PRT' ,'VERB' ,'X', '.']

vocab_size=56057

def welcome():
    return "Welcome All"

def ptag(word):
    global E
    L = []
    check = 0
    for i in range(len(E)):
        if word in E[i]:
            L.append(i)
            check = 1
    if check == 0:
        L = list(tagset.values())
        
    return L


"Viterbi Algorithm"

def preprocessing(sentence):
    for i in string.punctuation:
        if i in sentence:
            sentence=sentence.replace(i," "+i)
    return sentence

def Viterbi(sentence,k):
    if k == 0:
        word = sentence[k].lower()
        taglist = ptag(word)
        Prob = []
        Label = []
        for i in taglist:
            if word in E[i]:
                prob = T[0,i]*E[i][word]
            else:
                prob = T[0,i]/(sum(list(E[i].values()))+vocab_size)
            Prob.append(prob)
            Label.append(tagkeys[i])
            
        if k == len(sentence)-1:
            return Label[Prob.index(max(Prob))]
        
        return (Prob,Label)
           
    word = sentence[k].lower()
    taglist = ptag(word)
    prevprob,prevtag = Viterbi(sentence,k-1)
    Probj = []
    Labelj = []
    
    for i in taglist:
        maxi = 0
        add = 0
        for j in range(len(prevprob)):
            # print(j)
            # print(prevtag[j].split("__")[-1])

            A = prevprob[j]*T[tagset[prevtag[j].split("__")[-1]]+1,i]
            if A>= maxi:
                maxi = A
                add = i
                label = prevtag[j]+"__"+tagkeys[i]
        
        if word in E[add]:
            Probj.append(maxi*E[add][word])
        else:
            Probj.append(maxi/(sum(list(E[add].values()))+vocab_size))
#         except KeyError as ke:
#             print('Did not encounter ', ke, ' in the training corpus.')
        Labelj.append(label)
        
    if k == len(sentence)-1:
        return Labelj[Probj.index(max(Probj))]
    
    return (Probj,Labelj)

def predict(sentence):
    sentence = [i.lower() for i in sentence]
    return Viterbi(sentence,len(sentence)-1)


def main():
    st.title("POS Tagger")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit POS Tagging ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sentence = st.text_input("Sentence","Type Here")

    result=""
    if st.button("Predict"):
        tio=preprocessing(sentence)
        sr=tio.split(" ")
        print(sr)
        result=predict(sr)
        print(result)

    st.success("The Pos Tags are :",result.split("__"))

if __name__=='__main__':
    main()
    
    
    