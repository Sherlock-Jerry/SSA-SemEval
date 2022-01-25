import json
import os
import sys
import random

annotations = sys.argv[1]
results     = sys.argv[2]

files = os.listdir(annotations)

aspect_rel   = results+"/"+"relations_aspects.txt"
sentiment_rel= results+"/"+"relations_sentiments.txt"

train_sent   = results+"/train_bert.sent"
dev_sent     = results+"/dev_bert.sent"
test_sent    = results+"/test_bert.sent"

train_tuple  = results+"/train_bert.tuple"
dev_tuple    = results+"/dev_bert.tuple"
test_tuple   = results+"/test_bert.tuple"

train_pointer= results+"/train_bert.pointer"
dev_pointer  = results+"/dev_bert.pointer"
test_pointer = results+"/test_bert.pointer"

sentences    = []
all_tuples   = []
all_pointers = []

count = 0

relations = set()
aspect_types = set()
sentiments = set()

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
        if len(aspects) == 0:
            continue
        sentence = annotation["sentence"]
        sentence = remove_dot(sentence)

        tuples = []
        pointers = []
        #print(aspects)
        for aspect in aspects:
            aspect_phrase = aspect["aspect_phrase"]
            aspect_phrase = remove_dot(aspect_phrase)
            aspect_type = aspect["aspect_type"]
            opinion_info_phrase = aspect["opinion_info_phrase"]
            opinion_info_phrase = remove_dot(opinion_info_phrase)
            sentiment = aspect["sentiment"].upper()

            aspect_types.add(aspect_type)
            sentiments.add(sentiment)

            tokens = sentence.split()
            aspect_tokens = aspect_phrase.split()
            opinion_tokens = opinion_info_phrase.split()

            len_tokens = len(tokens)
            len_aspect_tokens = len(aspect_tokens)
            len_opinion_tokens = len(opinion_tokens)

            aspect_start = -1
            aspect_end   = -1
            opinion_start= -1
            opinion_end  = -1
            stake_start  = -1
            stake_end    = -1

            for i in range(len_tokens-len_aspect_tokens+1):
                flag = 1
                for j in range(len_aspect_tokens):
                    if tokens[i+j] != aspect_tokens[j]:
                        flag = 0
                        break
                if flag==1:
                    aspect_start = i+1
                    aspect_end   = i+len_aspect_tokens
                    break
            
            if aspect_start<=0:
                print("id={}".format(annotation['id']))
                print("aspect={}".format(aspect_phrase))
                print("opinion={}".format(opinion_info_phrase))
                continue
            
            for i in range(len_tokens-len_opinion_tokens+1):
                flag = 1
                for j in range(len_opinion_tokens):
                    if tokens[i+j] != opinion_tokens[j]:
                        flag = 0
                        break
                if flag==1:
                    opinion_start = i+1
                    opinion_end   = i+len_opinion_tokens
                    break

            if opinion_start<=0:
                print("id={}".format(annotation['id']))
                print("aspect={}".format(aspect_phrase))
                print("opinion={}".format(opinion_info_phrase))
                continue

            stakeholder = "[unused]"

            if "stakeholder" in aspect.keys():
                stakeholder = aspect["stakeholder"]
                stake_tokens= stakeholder.split()
                len_stake_tokens = len(stake_tokens)
                for i in range(len_tokens-len_stake_tokens+1):
                    flag = 1
                    for j in range(len_stake_tokens):
                        if tokens[i+j] != stake_tokens[j]:
                            flag = 0
                            break
                    if flag==1:
                        stake_start = i+1
                        stake_end   = i+len_stake_tokens
                        break
            else:
                stake_start = 0
                stake_end   = 0

            if stake_start<0:
                print("id={}".format(annotation['id']))
                print("stk")
                continue                        

            pointer = [str(aspect_start),str(aspect_end),aspect_type,str(opinion_start),str(opinion_end),
            sentiment,str(stake_start),str(stake_end)]
            pointer = ";".join(pointer)
            pointers.append(pointer)

            tuple   = [aspect_phrase,aspect_type,opinion_info_phrase,sentiment,stakeholder]
            tuple   = ";".join(tuple)
            tuples.append(tuple)
            count+=1
        
        
        if len(tuples)>0 and len(pointers)>0:
            sentences.append("[unused] "+sentence)
            tuples = "|".join(tuples)
            all_tuples.append(tuples)
            pointers = "|".join(pointers)
            all_pointers.append(pointers)

print(count)

length = len(sentences)
indices = [i for i in range(length)]
train_frac = 0.8
train_num = train_frac*length
train_indices = random.sample(indices,int(train_num))
rest_indices = [i for i in range(length) if i not in train_indices]
dev_frac = 0.5
dev_indices = random.sample(rest_indices,int(len(rest_indices)*dev_frac))
test_indices = [i for i in rest_indices if i not in dev_indices]

train_sentences = [sentences[i] for i in train_indices]
dev_sentences   = [sentences[i] for i in dev_indices]
test_sentences  = [sentences[i] for i in test_indices]

train_tuples    = [all_tuples[i] for i in train_indices]
dev_tuples      = [all_tuples[i] for i in dev_indices]
test_tuples     = [all_tuples[i] for i in test_indices]

train_pointers  = [all_pointers[i] for i in train_indices]
dev_pointers    = [all_pointers[i] for i in dev_indices]
test_pointers   = [all_pointers[i] for i in test_indices]

def annotate(filename,sequence):
    with open(filename,"w") as f:
        for line in sequence:
            f.write(line)
            f.write('\n')

annotate(train_sent,train_sentences)
annotate(dev_sent,dev_sentences)
annotate(test_sent,test_sentences)

annotate(train_tuple,train_tuples)
annotate(dev_tuple,dev_tuples)
annotate(test_tuple,test_tuples)

annotate(train_pointer,train_pointers)
annotate(dev_pointer,dev_pointers)
annotate(test_pointer,test_pointers)

annotate(aspect_rel,aspect_types)
annotate(sentiment_rel,sentiments)

