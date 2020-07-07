from google.cloud import translate_v2 as translate
import pandas as pd
import os
translate_client = translate.Client()

def translate_files_in_folder(arr):

    for file in arr:
        df = pd.read_csv(root_path + 'resources/'+file, encoding='ISO 8859-9')
        translation_column = []
        for t in df['transcript']:
           translation_column.append(translate(t, "en"))
        df['translation'] = translation_column
        df.to_csv(root_path+"resources/"+file+"_translated.csv")

    return

def translate(text, target):
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text, target_language=target)
    #print(u'Text: {}'.format(result['input']))
    #print(u'Translation: {}'.format(result['translatedText']))
    #print(u'Detected source language: {}'.format(
     #   result['detectedSourceLanguage']))
    return result['translatedText']

if __name__ == "__main__":
    root_path = "/Users/Gizem/PycharmProjects/bipolar-disorder-recognition/"
    arr = os.listdir(root_path + "resources")
    translate_files_in_folder(arr)

    ### TEST ###
    text = "seviyorum"
    target= "en"
    translate(text, target)
