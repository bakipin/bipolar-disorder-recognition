#!/usr/bin/python3
# coding: utf8

#%%
from affectModel import *
import csv
#%%
DIR_A='AffectResources'
CommonWords = os.path.join(DIR_A, 'inp.txt')
CommonWordsCorrected = os.path.join(DIR_A,'corrects.txt')
MekMak = os.path.join(DIR_A,'mek-mak_stem_list.csv')
AfDic=os.path.join(DIR_A,'AffectDictionary.csv')
AfDicMarked=os.path.join(DIR_A,'AffectDictionaryMarked.csv')

list1=get_to_list(CommonWords)
list2=get_to_list(CommonWordsCorrected)
dicc=zip_to_dict(list1,list2)
mekmak= get_to_list(MekMak)
normwords=get_input(AfDic)
markedwords=get_input(AfDicMarked)
onlywords=[item[0] for item in normwords]
normwords=remove_mekmak(normwords,mekmak)
markedwords=remove_mekmak(markedwords,mekmak)
affdic={item[0]:[float(item[2]),float(item[3]),float(item[4]),item[5]] for item in normwords}
markeddic={item[0]:[float(item[2]),float(item[3]),float(item[4]),item[5]] for item in markedwords}
affdic=vowel_harmony(affdic, markeddic)
personN=get_to_list(PersonNames)


def test(inputtext, affdic):

        #inputtext = input('Please enter text to see affective prediction: \n')
        inputtext=re.sub(r'\.([a-zA-Z])', r'. \1', inputtext)   #insert space after "."
        testing=inputtext.split()
        for i in range(0,len(testing)):
            if testing[i] not in personN:
                testing[i]=testing[i].lower()

        affdic=negate_sizsuz(testing,affdic)
        affect={}
        repp = arousal_reps(testing)
        upp = upper_case(testing) #check the letter case
        sentence=testing[:]
        testing=correction15000(dicc,testing)
        testing=remove_reps(testing,onlywords)
        sen_cleaned=testing[:]
        emotion=check_emoticon(sentence)
        affect=check_affect(affect,testing,affdic)
        affect=check_stopwords(affect)
        affect=check_personNames(affect)
        affect=check_intensifiers(affect,sen_cleaned)
        affect,nene =check_negation(affect,sen_cleaned,affdic)
        polarity=check_polarity(affect)
        polarity3=check_polarity_3(affect)
        valence,arousal,dominance=overall_average_all_pos_tweet(affect,sen_cleaned,nene,emotion)
        valence,arousal,dominance= check_interjections(affect, polarity3,valence,arousal,dominance)
        valence,arousal,dominance= other_features(valence, arousal, dominance,repp,upp, polarity3)



        print('input text:', inputtext)
        print('normalized input:', " ".join(testing))
        print('Word level analysis: ')
        for a in affect:
            print(a, affect[a])
        print('Overall:',' Valence: ', valence, ' Arousal:', arousal, ' Dominance:', dominance)
        print(' ')
        return affect

#%%
target_path = './VAD_Scores_Task_Based/'
with open('transcripts_preprocessed.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
                if line_count == 0:
                        line_count += 1
                        continue
                else:
                        with open(target_path + row[0][:-4] + '.csv', 'w') as writeFile:
                                writer = csv.writer(writeFile)
                                aff_dict = test(row[1], affdic)
                                for key,val in aff_dict.items():
                                        writer.writerow([key, val[0], val[1], val[2], val[3]])