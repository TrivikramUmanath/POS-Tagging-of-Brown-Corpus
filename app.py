

import numpy as np

import pandas as pd
import streamlit as st 
import json
from PIL import Image

#app=Flask(__name__)
#Swagger(app)

# pickle_in = open("classifier.pkl","rb")
# classifier=pickle.load(pickle_in)

f = open ('Prior_Prob.json', "r")
Prior_Prob = json.load(f)
f.close()

f = open ('Transition_Prob.json', "r")
Transition_Prob = json.load(f)
f.close()

f = open ('Emission_Prob.json', "r")
Emission_Prob = json.load(f) 
f.close()

def welcome():
    return "Welcome All"

def ViterbiPredict(sentenc,Prior_Prob,Emission_Prob,Transition_Prob): 
    #Testing
    All_Pos=list(Prior_Prob.keys())
    sentence=sentence.lower()
    sentence = sentenc.split(" ")
    T = len(sentence)
    State_Seq = []
    State_Prob = {}
    first_word = sentence[0]
    if first_word in Emission_Prob.keys():
        for tag in Emission_Prob[first_word].keys():
            prob = Prior_Prob[tag] * Emission_Prob[first_word][tag]
            State_Prob[tag] = prob
    else:
        for tag in Prior_Prob.keys():
            prob = Prior_Prob[tag] * 1e-6  # Use a small positive value for unknown words
            State_Prob[tag] = prob
    pos = max(State_Prob, key=State_Prob.get)
    State_Seq.append(pos)

    for word_position in range(1, T):
        new_State_Prob = {}
        for k in State_Prob.keys():
            word = sentence[word_position][0]
            prob = State_Prob[k]
            for transit in Transition_Prob[k].keys():
                emit_prob = Emission_Prob.get(word, {}).get(transit, 1e-6) # Use a small positive value for unknown words
                new_prob = prob * Transition_Prob[k][transit] * emit_prob
                if transit not in new_State_Prob.keys() or new_prob > new_State_Prob[transit]:
                    new_State_Prob[transit] = new_prob
        # Update State_Prob for the next word position
        State_Prob = new_State_Prob
        # Append most probable POS tag for the current word based on Viterbi
        pos = max(State_Prob, key=State_Prob.get)
        State_Seq.append(pos)
    return State_Seq

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
        result=ViterbiPredict(sentence,Prior_Prob,Emission_Prob,Transition_Prob)
        # result=predict_note_authentication(variance,skewness,curtosis,entropy)

    st.success('The Pos Tags are {}'.format(result))

if __name__=='__main__':
    main()
    
    
    