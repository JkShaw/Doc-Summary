#!/usr/bin/python

__author__ = 'uday kale'
__email__  = 'udaygkale@gmail.com'

#This code uses TextRank Algorithm to summarize the data of a file
#It also works for multiple documents
#The graph of text can be viewed as image too

import sys
import glob

sys.path.append('..')
sys.path.append('/usr/lib/graphviz/python/')
sys.path.append('/usr/lib64/graphviz/python/')

import gv
import nltk
import cv2
import operator

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from pygraph.classes.graph import graph
from nltk.corpus import stopwords
from pygraph.readwrite.dot import write


def rank_sentences(file_names):
    scores = dict()
    edges = dict()
    sentence_id = 0
    word_relation = graph()
    
    text=""
    for file_name in file_names:
        text += open(file_name).read()
        
    sentences = nltk.data.load('tokenizers/punkt/english.pickle').tokenize(text.strip()) #Extract Sentences

    for sentence in sentences:  #For every sentence
        sentence_id += 1
        previous_word = ''
        sentence = sentence.replace("\n", "").lower()
        word_relation.add_node(str(sentence_id)) #Add sentence number to the graph
        scores[str(sentence_id)] = 0.25
        edges[str(sentence_id)] = 0
        sentence = RegexpTokenizer(r'\w+').tokenize(sentence) #Split the sentence into words

        for present_word in sentence:
            if not present_word.isdigit():
                lemma = WordNetLemmatizer()
                lemma.lemmatize(present_word)
                #Check if present word is stop word
                if present_word not in stopwords.words('english') and present_word not in previous_word:
                    #Add a present word as node and words dictionary entry if it is not in the graph
                    if word_relation.has_node(present_word) is False:
                        word_relation.add_node(present_word)
                        scores[present_word] = 0.25 #Set initial score of all the words to 0.25
                        edges[present_word] = 0
                    #Add an edge between present word and previous word
                    if previous_word is not '' and word_relation.has_edge((present_word, previous_word)) is False:
                        word_relation.add_edge((present_word, previous_word))
                        edges[present_word] = edges[present_word] + 1
                        edges[previous_word] = edges[previous_word] + 1
                    #Add Edge between current sentence and present word
                    if not word_relation.has_edge((str(sentence_id), present_word)):
                        word_relation.add_edge((str(sentence_id), present_word))
                        edges[present_word] = edges[present_word] + 1
                        edges[str(sentence_id)] = edges[str(sentence_id)] + 1
                    previous_word = present_word
    
    new_scores = dict()
    i = 0
    
    while i < 15:
        i += 1
        for word in scores.keys():
            neghibors = word_relation.neighbors(word)
            value = 0    
            for neghibor in neghibors:
                value += scores[neghibor]/edges[neghibor]
                value *= 0.85
                value += 0.15
            new_scores[word] = value

        for word in new_scores.keys():
            scores[word] = new_scores[word]

    sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1))
    sorted_scores.reverse()
    print sorted_scores
    
    #print top sentences
    top=3
    i=0
    if top<sentence_id:
        for sentence_id, score in sorted_scores:
            if i<top:
                i+=1
                print score, sentences[int(sentence_id)]
                
    return word_relation

#Function to display an image
#open cv library routines are used
def display_image(image_name):
    img = cv2.imread(image_name)
    x, y = img.shape[1] / 8 , img.shape[0] / 2
    img = cv2.resize(img, (x, y))
    while 1:
        cv2.imshow('word_relation', img)
        key = cv2.waitKey(33)
        if key == 27:
            break
        else:
            continue


#put the proper path before execution
path="path to the samples directory files"
file_names=glob.glob(path)
word_relation=rank_sentences(file_names)

#create image of the graph
dot = write(word_relation)
gvv = gv.readstring(dot)
gv.layout(gvv, 'dot')
gv.render(gvv, 'png', 'word_relation.png')
#display the image
display_image('word_relation.png')

print '*****done*****'
