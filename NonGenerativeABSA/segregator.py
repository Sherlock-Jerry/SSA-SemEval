import json
import os
import sys
import random

annotations = sys.argv[1]
results     = sys.argv[2]

files = os.listdir(annotations)

pos_file   = results+"/positive.sent"
neg_file   = results+"/negative.sent"

pos_sentences    = []
neg_sentences    = []

pos_count = 0
neg_count = 0

def remove_dot(text):
    fg = 0
    ind = 0
    for i in reversed(range(len(text))):
        if text[i]==" ":
            continue
        elif text[i]==".":
            fg=1
            ind=i
            break
        else:
            break
    if fg==1:
        return text[:ind]+" "
    else:
        return text

for filename in files:
    print(filename)
    if filename[-5:] != ".json":
        continue
    file = open(annotations+"/"+filename)
    data = json.load(file)

    mainname = filename[:-5]
    
    for annotation in data:
        aspects = annotation["aspects"]
        sentence = annotation["sentence"]
        sentence = remove_dot(sentence)
        

        if len(aspects) == 0:
            neg_count+=1
            neg_sentences.append("[unused] "+sentence)
        else:
            pos_count+=1
            pos_sentences.append("[unused] "+sentence)

print("positive sentences = {}".format(pos_count))
print("negative sentences = {}".format(neg_count))

def annotate(filename,sequence):
    with open(filename,"w") as f:
        for line in sequence:
            f.write(line)
            f.write('\n')

annotate(pos_file, pos_sentences)
annotate(neg_file, neg_sentences)
