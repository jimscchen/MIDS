import re
import nltk
import numpy as np
import pandas as pd
import os
import json

personPron1 = ['i', 'me', 'my', 'mine', 'myself']
personPron1p = ['we', 'us', 'ours', 'our', 'ourselves']
personPron2 = ['you', 'your', 'yours', 'yourself']
personPron3m = ['he', 'his', 'him', 'himself']
personPron3f = ['she', 'her', 'hers', 'herself']
personPron3p = ['they', 'them', 'theirs', 'themselves']

def getRelations():
    return {0:'others', 1:'negative mentioning', 2:'positive mentioning', 3:'identity mentioning', 4: 'mixed mentioning', 5: 'place mentioning', 6:'same team mentioning', 7:'opposing team mentioning'}

def simpleRE(rows):
    '''
    takes in tokens and extract potential relations
    if 'char' is available for a token, use the value instead of the actual content
    '''
    relation = []
    nsubj = -1
    verb = -1
    for i, token in enumerate(rows.tokens):
        if token['label'] == 'NSUBJ':
            nsubj = i
            verb = token['index']
        if 'OBJ' in token['label'] and token['index'] == verb:
            subj = rows.tokens[nsubj].get('char', rows.tokens[nsubj]['content'])
            obj = rows.tokens[i].get('char', rows.tokens[i]['content'])
            relation.append({'relation':rows.tokens[verb]['content'], 
                             'ent1':subj, 'ent2':obj, 'class':0, 'line':rows.name})
        
    if relation:
        return relation
    else:
        return None
    
def extract_relation_categories(cList, rows):
    relation = []
    
    #relation += extract_mention_team(rows)
    relation += extract_place_mentioned(rows)
    relation += extract_mention_sentiment(cList, rows)
    relation += extract_identity(cList, rows)
    
    if relation:
        return relation
    else:
        return None    

def extract_identity(cList, rows):
    relation = []
    for entity in rows.entities:
        if entity['name'] in cList:
            for mention in entity['mentions']:
                if mention.lower() not in personPron1 + personPron1p + personPron2 + \
                personPron3m + personPron3f + personPron3p and mention.lower() not in entity['name'].lower():
                    relation.append({'relation':'character ' + entity['name'] + ' has identity of ' + mention, 'ent1':entity['name'],
                                     'ent2':entity['name'], 'class': 3, 'line':rows.name, 'men2':entity['mentions']})
    return relation
    
def extract_mention_team(rows):
    if rows.speaker == 'narrator':
        return []
 
    relation = []
    for token in rows.tokens:
        if token['lemma'].lower() in personPron1p:
            for char in token['char']:
                if char != rows.speaker:
                    relation.append({'relation':'belong to same team', 'ent1':rows.speaker, 'ent2':char, 'class':6, 'line':rows.name,
                                     'men2':[token['content']]})

        if token['lemma'].lower() in personPron3p:
            if rows.sentiment['score'] < - 0.4:
                for char in token['char']:
                    relation.append({'relation':'belong to opposing team', 'ent1':rows.speaker, 'ent2':char, 'class':7, 'line':rows.name,
                                     'men2':[token['content']]})

            numChar = len(token['char'])
            for i in range(numChar):
                for j in range(i+1, numChar):
                    relation.append({'relation':'belong to same team', 'ent1':token['char'][i], 'ent2':token['char'][j], 
                                     'class':6, 'line':rows.name, 'men2':[token['content']]})
                
    return relation
        
def extract_place_mentioned(rows):
    if rows.speaker == 'narrator':
        return []
    relation = []
    places_list = [e for e in rows.entities if e['type'] == 'LOCATION']
    
    for p in places_list:
        rel_index = -1
        place_is_subj = False
        
        for i, token in enumerate(rows.tokens):
            rel_phrase = ''
            if token['label'] == 'NSUBJ' and token['content'] in p['name'].split(' '):
                nsubj = i
                rel_index = token['index']
                place_is_subj = True
                
            if token['label'] == 'NSUBJ':
                nsubj = i
                rel_index = token['index']
                
            if 'OBJ' in token['label'] and token['index'] == rel_index and token['content'] in p['name'].split(' '):
                subj = rows.tokens[nsubj].get('char', [rows.tokens[nsubj]['content']])[0]
                obj = p['name']
                
                for j in range(rel_index, i):
                    rel_phrase += ' ' + rows.tokens[j]['content']
                relation.append({'relation': rel_phrase, 
                                 'ent1':subj, 'ent2':obj, 'class':5, 'line':rows.name})
                
            if 'OBJ' in token['label'] and token['index'] == rel_index and place_is_subj:
                subj = p['name']
                obj = rows.tokens[i].get('char', [rows.tokens[i]['content']])[0]
                
                for j in range(rel_index, i):
                    rel_phrase += ' ' + rows.tokens[j]['content']
                relation.append({'relation': rel_phrase, 
                                 'ent1':subj, 'ent2':obj, 'class':5, 'line':rows.name})
        
    return relation
    
def extract_mention_sentiment(cList, rows):
    if rows.speaker == 'narrator':
        return []
    relation = []
    persons_list = [e for e in rows.entities if e['type'] == 'PERSON' and e['name'] != rows.speaker and e['name'] in cList]
    sentiment_score = rows.sentiment['score']
    sentiment_mag = rows.sentiment['magnitude']
    
    for e in persons_list:
    
        if sentiment_score > 0.4:
            relation.append({'relation': rows.dialogue, 
                             'ent1':rows.speaker, 'ent2': e['name'], 'class': 2, 'line':rows.name, 'men2':e['mentions']})
    
        elif sentiment_score < -0.4:
            relation.append({'relation': rows.dialogue, 
                             'ent1':rows.speaker, 'ent2': e['name'], 'class': 1, 'line':rows.name, 'men2':e['mentions']})
                         
        elif sentiment_mag > 1.0:
            relation.append({'relation': rows.dialogue, 
                             'ent1':rows.speaker, 'ent2': e['name'], 'class': 4, 'line':rows.name, 'men2':e['mentions']})
    return relation

def REEval(dfList, numExamples=50):
    '''
    This function takes a list of dfs and evaluate model performance
    Enhance the original google api df with one of the model functions
    '''
    numModels = len(dfList)
    sampled = np.zeros(numModels)
    correct = np.zeros(numModels)
    
    # indexes for lines of dialogue with resolved pronouns
    df = dfList[0]
    REIndex = list(df[df.relations.notnull()].index)
    
    # sample lines
    selectLine = np.random.choice(REIndex, numExamples, replace=False)
    for lineNum in selectLine:

        # for each model df, select line to analyze
        for m in range(numModels):
            
            # select model results
            df = dfList[m]

            # print line being analyzed
            print('\n' + '*'*8 + ' line {} '.format(lineNum) + '*'*8)
            for rowNum in range(max(0, lineNum - 2), min(len(dfList[m]), lineNum + 3)):
                if rowNum == lineNum:
                    print('=> {}. {}:\n=> {}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))
                else:
                    print('{}. {}:\n{}\n'.format(rowNum, df.loc[rowNum]['speaker'], df.loc[rowNum]['dialogue']))

            # print resolved pronouns from model
            print('*'*8 + ' test model {}: line {} '.format(m+1, lineNum) + '*'*8)
            print('{} relations identified'.format(len(df.loc[lineNum]['relations'])))
            for relation in df.loc[lineNum]['relations']:
                print('entities: {} => {}-{}'.format(relation['ent1'], relation['ent2'], relation.get('men2')))
                print('relation: {}'.format(relation['relation']))
                print('category: {}. {}'.format(relation['class'], getRelations()[relation['class']]))
                      

            collectInput = False

            # prompt user for count of correctly resolved pronouns
            while not collectInput:
                try:
                    count = int(input('\nhow many are correctly identified? '))
                    collectInput = True
                except:
                    print('incorrect input, only numbers allowed')

            # update counts of lines sampled from script and correctly extract relations
            sampled[m] += len(df.loc[lineNum]['relations'])
            correct[m] += count
        
    # calculate and print precision for all models
    print('\n' + '*'*8 + ' test results ' + '*'*8)
    for i, modelResult in enumerate(zip(correct, sampled)):
        if modelResult[0] == 0:
            result = 0
        else:
            result = modelResult[0]/modelResult[1]
        print('model %i: precision = %.2f (%i/%i correct)'%(i+1, result, correct[i], sampled[i]))