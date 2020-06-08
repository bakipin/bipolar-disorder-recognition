# -*- coding: utf-8 -*-

#                            .-"""-.
#                           / .===. \
#                          / / a a \ \
#                         / ( \___/ ) \
#       _______________ooo\__\_____/__/__________________
#      /                                                 \
#     |     Tools for a lexicon-rule based affect and     |
#     |   sentiment analysis in Turkish language.         |
#     |                                                   |
#     |    Author: Eda Aydin Oktay                        |
#      \_______________________________ooo_______________/
#                        /           \
#                       /:.:.:.:.:.:.:\
#                           |  |  |
#                           \==|==/
#                           /-'Y'-\
#                          (__/ \__)


import itertools #iterators for efficient looping
import re # regular expressions
import copy
import string
import os
import unidecode


DIR_A='AffectResources'
IntensifierTable = os.path.join(DIR_A, 'IntensifierTa.txt')
InterjectionTable = os.path.join(DIR_A,'interjectionTaE.txt')
EmoticonTable = os.path.join(DIR_A,'EmoticonT.txt')
Stopwords = os.path.join(DIR_A,'stopwords.csv')
PersonNames= os.path.join(DIR_A,'personN.csv')



#get input in truecase
def get_input(filename):
    data=open(filename, 'r', encoding='utf-8')
    liste=data.readlines()  
    utterlist=[item.strip().split(",") for item in liste] 
    return utterlist


#get input in lowercase    
def get_input_lc(filename):
    data=open(filename, 'r', encoding='utf-8')
    liste=data.readlines()  
    utterlist=[item.lower().strip().split() for item in liste] 
    return utterlist


#get input in lowercase, punctuations are removed 
def get_input_lcw(filename):
    data=open(filename, 'r', encoding='utf-8')
    liste=data.readlines()  
    utterlist=[item.translate(str.maketrans('','', string.punctuation)).strip().split() for item in liste] 
    return utterlist

def get_to_list(inputfile):
    data=open(inputfile, 'r', encoding='utf-8')
    liste=data.readlines() 
    inputlist=[item.strip() for item in liste] 
    return inputlist


def readTodict(filename):
    with open(filename,'r', encoding='utf-8') as text:
        return dict(line.strip().split() for line in text)
 
 
#combine list to a dictionary
def zip_to_dict(list1,list2):
        comb = zip(list1,list2)    
        dicc={}
        for list1,list2 in comb:
           dicc[list1] = list2
        return dicc


#check manually corrected top word dictionary
def correction15000(dicc,wlist):
    for i in range(0,len(wlist)):
        if wlist[i] in dicc:
            wlist[i]=dicc[wlist[i]]
    return wlist


  
 #check presence of repetitive chars          
def arousal_reps(sentence):
    sent= ' '.join(sentence)
    rep=False
    checkrep=[[k,len(list(g))] for k, g in itertools.groupby(sent)]
    for letter in checkrep:
        if letter[1] >= 3:
            rep=True
    return rep        



#generate adjectives with -sHz suffix
def negate_sizsuz(testing,affdic):    
    without=["sız", "siz","suz","süz"]        
    for morph in without:
        for word in testing:
            if word[-3:] == morph:
                if word not in affdic:
                    if word[:-3] in affdic:
                         stem= copy.deepcopy(word[:-3])                        
                         affdic[word]=[3,3,3,'NN']
                         affdic[word][0] = 6- affdic[stem][0]
                         affdic[word][1] = 6- affdic[stem][1]
                         affdic[word][2] = 6- affdic[stem][2]
                         affdic[word][3] = affdic[stem][3]
    return affdic                         
 

#generate words from marked dictionary with vowel harmony/consonant change
# note: only changes on stem are marked
def vowel_harmony(affdic, markeddic):
    for word in markeddic:
            convert=''
          
            if word[-2:] == "k^":               
                convert=word[:-2]+'ğ'
                                
            if word[-2:] == "p^":
                convert=word[:-2]+'b'
          
            if word[-2:] == "t^":      
                convert=word[:-2]+'d'
       
            if word[-2:] == "g^":
                convert=word[:-2]+'ğ'
                
            if word[-2:] == "a^":
                convert=word[:-2]+'u'
                
            if word[-2:] == "a>":
                convert=word[:-2]+'ı'  
                
            if word[-2:] == "e>":
                convert=word[:-2]+'i'
                
            if word[-2:] == "e^":
                convert=word[:-2]+'ü' 
                
            if word[-2:] == "c^":
            
                convert=word[:-2]+'c'

            if "<" not in word:
                if len(convert)!=0:
                    
                    
                    affdic[convert]=[3,3,3,'NN']
                    affdic[convert][0]= markeddic[word][0]
                    affdic[convert][1]= markeddic[word][1]
                    affdic[convert][2]= markeddic[word][2]
                    affdic[convert][3]= markeddic[word][3]

    for item in markeddic:
            convert=''
            let= unidecode.unidecode(item)   #unicode(item,'utf-8')            
            if "<" in item:
                if "^" not in item:                                                       
                    convert= let[:let.index("<")]+let[-1] #vowel drop
                    convert= convert.encode('utf-8')
            if len(convert)!=0:                   
                    affdic[convert]=[3,3,3,'NN']
                    affdic[convert][0]= markeddic[item][0]
                    affdic[convert][1]= markeddic[item][1]
                    affdic[convert][2]= markeddic[item][2]
                    affdic[convert][3]= markeddic[item][3]
            
    return affdic
     
     
                            
#check the presence of any negation          
def check_negation(affect,sentence,affdic):  
       
       nene=False
       phrases=[phrase for phrase in affect]
       affcopy=copy.deepcopy(affect)
       inputsen=' '.join(sentence)
       if sentence.count("ne") >=2:
          nene=True #notr overall affect 
#       if ''.join(sentence).count("değil") >=2:           
#               nene=True     

       if nene != True:       
           for seg in affcopy:           
               if "değil" in seg: 
                  degil=seg
                  if len(sentence)!=1:  
                     nextword="dummy" 
                     if len(sentence)>=sentence.index(degil)+2:
                         
                        nextword=sentence[sentence.index(degil)+1]
                     if nextword != "mi":  #exclude "degil mi" 
                            preword=sentence[sentence.index(degil)-1] #word that "değil" negates
                            for ph in phrases:
                                if preword in ph:
                                    connected=copy.deepcopy(ph)
                            newword= connected+" "+"değil"
                            affect[newword]=[3,3,3,'NAN']
                            affect[newword][0]= 6-affcopy[connected][0] # update valence
                            affect[newword][1]= 6-affcopy[connected][1] # update arousal
                            affect[newword][2]= 6-affcopy[connected][2] # update dominance
                            if preword == "yok":
                                affect[newword][3]='NAN' #neutrilize "yok degil"
                            else:   
                                affect[newword][3]='JPH' #ADJ phrase
                            if connected in affect:
                               del affect[connected]
                            if seg in affect:
                               del affect[seg]
  
           
           yok="yok"
           for seg in affcopy:           
               if "yok" == seg: 
                  yok=seg
                  if "yok değil" not in inputsen:
                     if len(sentence)!=1:                           
                        preword=sentence[sentence.index(yok)-1] #word that "yok" negates
                        for ph in phrases:
                            if preword in ph:
                                connected=copy.deepcopy(ph)
                        newword= connected+" "+"yok"                        
                        affect[newword]=[3,3,3,'NAN']
                        affect[newword][0]= 6-affcopy[connected][0] # update valence
                        affect[newword][1]= 6-affcopy[connected][1] # update arousal
                        affect[newword][2]= 6-affcopy[connected][2] # update dominance
                        affect[newword][3]='VPH' #ADJ phrase
                        if connected in affect:
                           del affect[connected]  
                        if seg in affect:
                           del affect[seg]


           affcopy=copy.deepcopy(affect)
           neg=["miyor", "mıyor", "muyor", "müyor","mayacak", "meyecek","meyeceğiz","mayacağız","mayacağınız ","meyeceğiniz ","meyeceğim","mayacağım","mayın","meyin"]                                           
           for seg in affcopy:    
              for morp in neg:
                 if morp in seg: 
                      affect[seg][0]= 6- affect[seg][0]  # update valence
                      affect[seg][1]= 6- affect[seg][1]   # update arousal
                      affect[seg][2]= 6- affect[seg][2]   # update dominance
                               
           neg=["madı", "medi", "memiş", "mamış","memeli","mamalı","madan","meden","mese","masa","memesi","maması"]                          
           for seg in affcopy:    
              for morp in neg:
                 if seg not in affdic:
                    if morp in seg: 
                      affect[seg][0]= 6- affect[seg][0]  # update valence
                      affect[seg][1]= 6- affect[seg][1]   # update arousal
                      affect[seg][2]= 6- affect[seg][2]   # update dominance                       
                      
           neg=["masın", "mesin","mem","mam","mez","mazdı","mezdi","mazsa","mezse","mazdınız","mezdiniz","mazdım","mezdim","mazmış","mezmiş","maz","mayız","meyiz","memek","mamak","meyen","mayan"]                          
           for seg in affcopy:
              seg2 = seg.translate(str.maketrans('','',',?!.'))
              for morp in neg:                
                 if seg2 not in affdic:
                    if morp == seg2[-len(morp):]: #check only last suffix
#                    if morp in seg2:    
                      affect[seg][0]= 6- affect[seg][0]  # update valence
                      affect[seg][1]= 6- affect[seg][1]   # update arousal
                      affect[seg][2]= 6- affect[seg][2]   # update dominance                       
            


       return affect, nene





#check the presence of any intensifier, update VAD scores
def check_intensifiers(affect,sentence):
    data=open(IntensifierTable, 'r', encoding='utf-8')
    liste=data.readlines()  
    intlist=[item.strip().split(',') for item in liste]  
    affcopy=copy.deepcopy(affect)
    phrases=[phrase for phrase in affcopy]
    

    
    for seg in affcopy:
        for inten in intlist:         
            if inten[0] == seg:                  
                if sentence[-1]!=seg:
                   
                    nextword=sentence[sentence.index(seg)+1] #word that modifier is connected
                    connected=copy.deepcopy(nextword)
                    for ph in phrases:
                        if nextword in ph:
                            connected=copy.deepcopy(ph)
                    if connected in affcopy:      #eklendi  
                      if affcopy[connected][3] != 'NAN':                        
                        newword= inten[0] +" "+connected   
                        if affcopy[connected][0]>3.0: #positive polarity
                            affect[newword]=[3,3,3,'NAN']
                            affect[newword][0]= affcopy[connected][0]+float(inten[1]) # update valence
                            affect[newword][1]= affcopy[connected][1]+float(inten[3]) # update arousal
                            affect[newword][2]= affcopy[connected][2]+float(inten[5]) # update dominance
                            affect[newword][3]='JPH' #ADJ phrase
                            if connected in affect:
                              if inten[0] in affect:
                                 if inten[0] != connected:
                                    map(affect.pop, [connected,inten[0]])
                                    
                           
                        if affcopy[connected][0]<3.0: #negative polarity
                            affect[newword]=[3,3,3,'NAN']
                            affect[newword][0]= affcopy[connected][0]+float(inten[2]) # update valence
                            affect[newword][1]= affcopy[connected][1]+float(inten[4]) # update arousal
                            affect[newword][2]= affcopy[connected][2]+float(inten[6]) # update dominance                    
                            affect[newword][3]='JPH' #ADJ phrase
                            if connected in affect:
                              if inten[0] in affect:
                                  if inten[0] != connected:
                                     map(affect.pop, [connected,inten[0]])

    return affect
           


#check the presence of any intensifier, update VAD scores
def check_interjections(affect, polarity, valence, arousal, dominance):
    data=open(InterjectionTable, 'r', encoding='utf-8')
    liste=data.readlines()  
    intlist=[item.strip().split(',') for item in liste]  
    
    for seg in affect:    
        for inten in intlist:
            if inten[0] == seg:                                
                    if polarity ==1 : #positive polarity
                       valence += float(inten[1])
                       arousal += float(inten[3])
                       dominance += float(inten[5])
                       
                    if polarity == -1: #negative polarity
                       valence += float(inten[2])
                       arousal += float(inten[4])
                       dominance += float(inten[6])


# check ? and !
    question=False
    exclam=False
    for seg in affect:
        if "?" in seg:
           question=True
        if "!" in seg:
           exclam=True

    if question==True:
       
        if polarity == 1 :
            valence += 0.2
            arousal += -0.1
            dominance += 0
        if polarity == -1 :
            valence += 0.1
            arousal += 0.8
            dominance += -0.6
        if polarity ==0 :
            valence += 0.2
            arousal += -1
            dominance += -0.1

    if exclam==True:
        
        if polarity == 1 :
            valence += -0.2
            arousal += 1.4
            dominance += 0.1



    return valence, arousal, dominance
                       

#discard initial compound sentence in sentiment analysis
def check_conjuction(sentence):
    sen= ' '.join(sentence)
    supdated=sentence
    conj=["ama","rağmen","fakat","oysa", "lakin", "yinede","dışında"]
    for co in conj:
        if co in sentence:
          if supdated[-len(co):] !=co:
            supdated=sen.split(co,1)[1]
            if len(supdated) > 1:
               supdated=supdated.split()                     
    return supdated


 
#check emoticon usage, calculate emoticon values
def check_emoticon(sentence):
    data=open(EmoticonTable, 'Ur', encoding='utf-8')
    liste=data.readlines()  
    emolist=[item.strip().split() for item in liste] 
    emotion=[]
    for word in sentence:
        for emo in emolist:
            if emo[0] in word:
                emotion.append(emo[1])
    return emotion
    

#check proper names of persons
def check_personNames(affect):
    personN=get_to_list(PersonNames)    
    affcopy=copy.deepcopy(affect)
    for unit in affcopy:
      for name in personN:         
         if unit==personN:               
            affect[unit]=[3,3,3,'NAN']    
    return affect
    
#stopwords neutralize stopwords
def check_stopwords(affect):
    stopWordList=get_to_list(Stopwords)
    affcopy=copy.deepcopy(affect)
    for unit in affcopy:
      if unit in stopWordList:          
         affect[unit]=[3,3,3,'NAN']    
    return affect



#remove redundant features
def feature_reduction(affect,filename):
    featurelist=get_to_list(filename)
    affcopy=copy.deepcopy(affect)
    for unit in affcopy:
      if unit in featurelist:   
         for fea in featurelist:
           if unit == fea:
              affect[unit]=[3,3,3,'NAN']    
    return affect


# remove repetitive letters of a word "seelaaaaam"
# except the words that exist in database e.g. 'cennet' and 'cehennem' 
def remove_reps(wlist,onlywords):    
    for i in range(0,len(wlist)): 
        #reduce to at most two repetitions
        word= re.sub(r'(.)\1+', r'\1\1', wlist[i]) 
        found=False
        for item in onlywords: #check database for word
            if item==word[:len(item)]:                              
                found=True
                                                      
        if found==False:        
            wlist[i] = ''.join(ch for ch, _ in itertools.groupby(word))
                   
        else:
           wlist[i] = word  
    return wlist
        
        
#remove -mek +mak infinitive suffix
def remove_mekmak(normwords,mekmak):
    for item in normwords:       
        if item[0] not in mekmak: #check exceptitional nouns ending with mak-mek (hamak,mercimek etc.)
            if item[0][-3:]== ('mek'):
                item[0]=item[0][:-3]
            if item[0][-3:]== ('mak'):
                item[0]=item[0][:-3]
                
    return normwords

       
# generate trigrams, bigrams or unigrams
def ngrams(tokens, min_n, max_n):
    n_tokens = len(tokens)
    ph=[]
    for i in range(n_tokens):
        for j in range(i+min_n, min(n_tokens, i+max_n)+1):
             ph.append(tokens[i:j])
    return ph



def check_affect(affect,testing,affdic):
        #first check dictionary for trigrams
        trigram= ngrams(testing,3,3) 
        if len(trigram) != 0:
            for i in trigram:
                tri=' '.join(i) 
                overlap_max=0
                overlap=0
                vale=0
                aros=0
                domi=0
                pos=0               
                for item in affdic:
                    if "bytes" in str(type(item)):
                        item = item.decode("utf-8")
                    if item.count(' ')>1:  #only check trigrams
                         if  item in tri:                              
                                overlap=len(item)
                                if overlap >= overlap_max:                   
                                     vale=affdic[item][0]
                                     aros=affdic[item][1]
                                     domi=affdic[item][2]
                                     pos=affdic[item][3]
                                     overlap_max=overlap   
#                                     ww=item
                if vale != 0:
                    segment=[vale,aros,domi,pos]
                    affect[tri]=segment
#                    affect[ww]=segment
                    for word in i:     
                        if word in testing:
                           testing.remove(word)
                            
                            
        bigram= ngrams(testing,2,2)
        if len(bigram) != 0:
            for i in bigram:
                phrase=' '.join(i) 
                overlap_max=0
                overlap=0
                vale=0
                aros=0
                domi=0
                pos=0              
                for item in affdic:
                      if "bytes" in str(type(item)):
                          item = item.decode("utf-8")
                      if item.count(' ')==1:  #only check bigrams
                         if  item in phrase:                               
                                overlap=len(item)
                                if overlap >= overlap_max:                   
                                     vale=affdic[item][0]
                                     aros=affdic[item][1]
                                     domi=affdic[item][2]
                                     pos=affdic[item][3]
                                     overlap_max=overlap 

                if vale != 0:
                    segment=[vale,aros,domi,pos]
                    affect[phrase]=segment
                    for word in i:  
                        if word in testing:
                           testing.remove(word)
                        
        unigram= ngrams(testing,1,1)     
        for i in unigram:
#            phrase=' '.join(i) 
            word=i[0]
            overlap_max=0
            overlap=0
            vale=0
            aros=0
            domi=0
            pos=0         
            for item in affdic:  
                  if word[:2]==item[:2]:
#                  if item[0].count(' ')==0:  #only check bigrams    
                     if  item in word:
                            overlap=len(item)
                            if overlap >= overlap_max:                   
                                 vale=affdic[item][0]
                                 aros=affdic[item][1]
                                 domi=affdic[item][2]
                                 pos=affdic[item][3]
                                 overlap_max=overlap

            if vale != 0:
                segment=[vale,aros,domi,pos]
                affect[word]=segment

                
            else:
                segment=[3.0,3.0,3.0,"NAN"]
                affect[word]=segment            
        return affect



#check positive or negative polary of input sentence
def check_polarity(affect):
    polarity=1
    v=[]
    for seg in affect:                
        if affect[seg][3] !='NAN':
                    v.append(affect[seg][0])                   
    if len(v)!=0:
        valAve=sum(v)/len(v)    
        if valAve<3:
            polarity=-1
        if valAve>=3:   
            polarity=1
   
    return polarity
    

#check positive, negative or neutral class of input sentence
def check_polarity_3(affect):
    polarity=1
    v=[]
    for seg in affect:                
        v.append(affect[seg][0])      
                    
    if len(v)!=0:
        valAve=sum(v)/len(v)    

        if valAve<=2.95:
            polarity=-1
        if valAve>=3.05:   
            polarity=1
        if valAve<3.05 and valAve>2.95:
            polarity=0         

    return polarity    
    
    

    
#update VAD with repetititition and upper case annotation scores 
def other_features(val,aro,dom,repp,upp,polarity):
    if repp:
            
            if polarity==1:
               val += 0.18
               aro += 0.87
               dom += 0.4         
            if polarity==-1:
               val += -0.36
               aro += 2.27
               dom += 0.4           
            if polarity==0:
               val += -0.22
               aro += 1.2
               dom += 0.85          
    if upp:
            if polarity==1:
               val += 0.24
               aro += 1.11
               dom += 0.3            
            if polarity==-1:
               val += -0.9
               aro += 1.5
               dom += 0.56
    
    return val,aro,dom
     

#check upper case usage in sentence
def upper_case(sentence):
    upp=False
    count =0
    for word in sentence:
        count += len([letter for letter in word if letter.isupper()])
    if count>3:
        upp=True
    return upp
    

#put upper and lower limit for valence, arousal, dominance 
def put_limit(val,aro,dom):   
    if val > 5:
        val=5
    if aro > 5:
        aro=5
    if dom > 5:
        dom=5
    if val < 1:
        val=1
    if aro < 1:
        aro=1
    if dom < 1:
        dom=1    
    return val, aro, dom

#calculate the overall affective score of sentence by averaging
def overall_average(affect,sentence,nene,emotion):  
    valence=3.0
    arousal=3.0
    dominance=3.0
    val=[]
    aro=[]
    dom=[]
    emotion=map(float,emotion)
    
    if nene==False:
        if len(emotion)!=0:
            average=sum(emotion)/len(emotion)
            valence=average
            arousal=average
            dominance=average
            
        else:
            for seg in affect:                
                if affect[seg][3] !='NAN':
                    val.append(affect[seg][0])
                    aro.append(affect[seg][1])
                    dom.append(affect[seg][2])
                    
            if len(val)!=0:
                
                valence=sum(val)/len(val)
                arousal=sum(aro)/len(aro)
                dominance=sum(dom)/len(dom)        
          
    return valence, arousal, dominance
    
    
 #calculate the overall affective score of sentence by averaging repetitive words too
 # no pos information is considered   
def overall_average_all(affect,sentence,nene,emotion):  
    valence=3.0
    arousal=3.0
    dominance=3.0
    val=[]
    aro=[]
    dom=[]
    emotion=map(float,emotion)
    
    if nene==False:
        if len(emotion)!=0:
            average=sum(emotion)/len(emotion)
            valence=average
            arousal=average
            dominance=average
            
        else:
            for seg in affect:                
                if affect[seg][3] !='NAN':
                   if seg in sentence:
                      cnt=sentence.count(seg) 
                   else:
                      cnt=1
                   for i in range(0,cnt):
                        val.append(affect[seg][0])
                        aro.append(affect[seg][1])
                        dom.append(affect[seg][2])
                    
            if len(val)!=0:
                
                valence=sum(val)/len(val)
                arousal=sum(aro)/len(aro)
                dominance=sum(dom)/len(dom)        
          
    return valence, arousal, dominance
    
#calculate the overall affective score of sentence by averaging repetitive words too
def overall_average_all_pos(affect,sentence,nene,emotion):  
    affcopy=copy.deepcopy(affect)
    phrases=[phrase for phrase in affcopy]
    valence=3.0
    arousal=3.0
    dominance=3.0
    val=[]
    aro=[]
    dom=[]
    emotion=map(float,emotion)
    
    if nene==False:
        if len(emotion)!=0:
            average=sum(emotion)/len(emotion)
            valence=average
            arousal=average
            dominance=average
            
        else:
            #neuralize NN with NN+VB structure
            for seg in affcopy: 
                if affect[seg][3]=="NN":                  
                   lastwordofunit=seg.split()[-1]            
                   for words in sentence:
                          if lastwordofunit in words:
                              lastunit=copy.deepcopy(words)
                           
                              
                   unitindex=sentence.index(lastunit)
                   if len(sentence)>=unitindex+2:
                      nextword=sentence[unitindex+1]    
                      nextunit="dummy"
                      for ph in phrases:
                          if nextword in ph:
                              nextunit=copy.deepcopy(ph)
                      if nextunit!="dummy":
                          posnext=affcopy[nextunit][3]
                          if posnext=="VB":
                             affect[seg][3]='NAN'
        
           
            for seg in affect:  
                if affect[seg][3] !='NAN':
                   if affect[seg][0] > 3 or affect[seg][0] < 2.99: #movierev. neutral range
                       if seg in sentence:
                          cnt=sentence.count(seg) 
                       else:
                          cnt=1
                       for i in range(0,cnt):
                            val.append(affect[seg][0])
                            aro.append(affect[seg][1])
                            dom.append(affect[seg][2])
                    
            if len(val)!=0:
                
                valence=sum(val)/len(val)
                arousal=sum(aro)/len(aro)
                dominance=sum(dom)/len(dom)        
          
    return valence, arousal, dominance   
 
#specific to tweets   
#calculate the overall affective score    
def overall_average_all_pos_tweet(affect,sentence,nene,emotion):  
    affcopy=copy.deepcopy(affect)
    phrases=[phrase for phrase in affcopy]
#    inputsen=' '.join(sentence)
    sentence = list(sentence)
    valence=3.0
    arousal=3.0
    dominance=3.0
    val=[]
    aro=[]
    dom=[]
    emotion=map(float,emotion)
    emotion=list(emotion)
    if nene==False:
        if len(emotion)!=0:
            average=sum(emotion)/len(emotion)
            valence=average
            arousal=average
            dominance=average
            
        else:
            #neuralize NN with NN+VB structure
            for seg in affcopy: 
                if affect[seg][3]=="NN":
                  
                   lastwordofunit=seg.split()[-1] 
           
                   for words in sentence:
                          if lastwordofunit in words:
                              lastunit=copy.deepcopy(words)
                           
                              
                   unitindex=sentence.index(lastunit)
                   if len(sentence)>=unitindex+2:
                      nextword=sentence[unitindex+1]    
                      nextunit="dummy"
                      for ph in phrases:
                          if nextword in ph:
                              nextunit=copy.deepcopy(ph)
                      if nextunit!="dummy":
                          posnext=affcopy[nextunit][3]
                          if posnext=="VB":
                             affect[seg][3]='NAN'
                             
            #neutralize word before "mi"
            for seg in affcopy: 
                 if seg =="mi" or seg =="mı":     
                    preword=sentence[sentence.index(seg)-1] #word that "mi" connected
                    for ph in phrases:
                        if preword in ph:
                            connected=copy.deepcopy(ph) 

                            affect[connected][3]='NAN'
                             
                             
           #neutralize NN with NN+ADJ structure
            for seg in affcopy: 
                if affect[seg][3]=="NN":                  
                   lastwordofunit=seg.split()[-1] 
                   for words in sentence:
                          if lastwordofunit in words:
                              lastunit=copy.deepcopy(words)
                                                        
                   unitindex=sentence.index(lastunit)
                   if len(sentence)>=unitindex+2:
                      nextword=sentence[unitindex+1]    
                      nextunit="dummy"
                      for ph in phrases:
                          if nextword in ph:
                              nextunit=copy.deepcopy(ph)
                      if nextunit!="dummy":
                          posnext=affcopy[nextunit][3]
                          if posnext=="ADJ" or posnext=="JPH" :
                             affect[seg][3]='NAN'             
               
           
            for seg in affect:  
                if affect[seg][3] !='NAN':
                    if affect[seg][0] > 3.5 or affect[seg][0] < 2.95:
                       if seg in sentence:
                          cnt=sentence.count(seg) 
                       else:
                          cnt=1
                       for i in range(0,cnt):
                            val.append(affect[seg][0])
                            aro.append(affect[seg][1])
                            dom.append(affect[seg][2])
                        
            if len(val)!=0:
                
                valence=sum(val)/len(val)
                arousal=sum(aro)/len(aro)
                dominance=sum(dom)/len(dom)        
          
    return valence, arousal, dominance    
    


def overall_average_all_pos_reh(affect,sentence,nene,emotion):  
    affcopy=copy.deepcopy(affect)
    phrases=[phrase for phrase in affcopy]
#    inputsen=' '.join(sentence)
    sentence=list(sentence)
    valence=3.0
    arousal=3.0
    dominance=3.0
    val=[]
    aro=[]
    dom=[]
    emotion=map(float,emotion)
    emotion=list(emotion)
    if nene==False:
        if len(emotion)!=0:
            average=sum(emotion)/len(emotion)
            valence=average
            arousal=average
            dominance=average
            
        else:
            #neuralize NN with NN+VB structure
            for seg in affcopy: 
                if affect[seg][3]=="NN":
                  
                   lastwordofunit=seg.split()[-1] 
           
                   for words in sentence:
                          if lastwordofunit in words:
                              lastunit=copy.deepcopy(words)
                           
                              
                   unitindex=sentence.index(lastunit)
                   if len(sentence)>=unitindex+2:
                      nextword=sentence[unitindex+1]    
                      nextunit="dummy"
                      for ph in phrases:
                          if nextword in ph:
                              nextunit=copy.deepcopy(ph)
                      if nextunit!="dummy":
                          posnext=affcopy[nextunit][3]
                          if posnext=="VB":
                             affect[seg][3]='NAN'
        
            for seg in affcopy: 
                 if seg =="mi" or seg =="mı":     
                    preword=sentence[sentence.index(seg)-1] #word that "mi" connected
                    for ph in phrases:
                        if preword in ph:
                            connected=copy.deepcopy(ph) 

                            affect[connected][3]='NAN'
                             
                             
           #neutralize NN with NN+ADJ structure
            for seg in affcopy: 
                if affect[seg][3]=="NN":                  
                   lastwordofunit=seg.split()[-1] 
                   for words in sentence:
                          if lastwordofunit in words:
                              lastunit=copy.deepcopy(words)
                                                        
                   unitindex=sentence.index(lastunit)
                   if len(sentence)>=unitindex+2:
                      nextword=sentence[unitindex+1]    
                      nextunit="dummy"
                      for ph in phrases:
                          if nextword in ph:
                              nextunit=copy.deepcopy(ph)
                      if nextunit!="dummy":
                          posnext=affcopy[nextunit][3]
                          if posnext=="ADJ" or posnext=="JPH" :
                             affect[seg][3]='NAN'             
             
           
            for seg in affect:  
                if affect[seg][3] !='NAN':
                    if affect[seg][0] > 3.5 or affect[seg][0] < 2.95:
                       if seg in sentence:
                          cnt=sentence.count(seg) 
                       else:
                          cnt=1
                       for i in range(0,cnt):
                            val.append(affect[seg][0])
                            aro.append(affect[seg][1])
                            dom.append(affect[seg][2])
                    
            if len(val)!=0:
                
                valence=sum(val)/len(val)
                arousal=sum(aro)/len(aro)
                dominance=sum(dom)/len(dom)        
          
    return valence, arousal, dominance    
    
    
    
    
    
#calculate the overall affective score of sentence by minmax method
def overall_minmax(affect,sentence,nene,emotion):
    valence=3.0
    arousal=3.0
    dominance=3.0
    val=[]
    aro=[]
    dom=[]
    emotion=map(float,emotion)  
    if nene==False:
        if len(emotion)!=0:
            average=sum(emotion)/len(emotion)
            valence=average
            arousal=average
            dominance=average
            
        else:
            for seg in affect:
                if affect[seg][3]=='JPH':
                    val.append(affect[seg][0])
                    aro.append(affect[seg][1])
                    dom.append(affect[seg][2])
                    break
                if affect[seg][3]=='ADJ':
                    val.append(affect[seg][0])
                    aro.append(affect[seg][1])
                    dom.append(affect[seg][2])
                    
                if affect[seg][3]=='VB':
                    val.append((1.1)*affect[seg][0])
                    aro.append(affect[seg][1])
                    dom.append(affect[seg][2])
                    
                if affect[seg][3]=='NN':
                    val.append(affect[seg][0])
                    aro.append(affect[seg][1])
                    dom.append(affect[seg][2])
                    
                if len(val)==0:   
                   if affect[seg][3] !='NAN':
                      val.append(affect[seg][0])
                      aro.append(affect[seg][1])
                      dom.append(affect[seg][2])
                      
            if len(val)!=0:
                maxv=max(val)
                minv=min(val)
                maxa=max(aro)
                mina=min(aro)
                maxd=max(dom)
                mind=min(dom)
                
                if abs(maxv-3)>=abs(minv-3):
                    vv=maxv
                else:
                    vv=minv
                    
                if abs(maxa-3)>=abs(mina-3):
                    aa=maxa
                else:
                    aa=mina
                    
                if abs(maxd-3)>=abs(mind-3):
                    dd=maxd
                else:
                    dd=mind   
                valence=vv
                arousal=aa
                dominance=dd      
    
    return valence, arousal, dominance
