import numpy as np
import json
import pprint as pp

from anytree import RenderTree
from zss import *

from retrieve_recipes import *
from im2recipe_w2v_weight import *

recipes = get_recipes_json('chocochip') #list

def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.label))

        
#pairwise Jaccard Sim

from operator import add

import re
import json
import nltk
from NYTtagger.lib.training import utils
from string import punctuation

import pycrfsuite
tagger = pycrfsuite.Tagger()
tagger.open('NYTtagger/tmp/trained_pycrfsuite')


from nltk.tokenize import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()

def sent2labels(sent):
    return [word[-1] for word in sent]

def sent2features(sent):
    return [word[:-1] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]   

def get_sentence_features(sent):
    """Gets  the features of the sentence"""
    sent_tokens = list(utils.tokenize(utils.cleanUnicodeFractions(sent)))

    sent_features = []
    for i, token in enumerate(sent_tokens):
        token_features = [token]
        token_features.extend(utils.getFeatures(token, i+1, list(sent_tokens)))
        sent_features.append(token_features)
    return sent_features

def format_ingredient_output(tagger_output, display=False):
    """Formats the tagger output into a more convenient dictionary"""
    data = [{}]
    display = [[]]
    prevTag = None

    for token, tag in tagger_output:
    # turn B-NAME/123 back into "name"
#        tag = re.sub(r'^[BI]\-', "", tag).lower()

        # ---- DISPLAY ----
        # build a structure which groups each token by its tag, so we can
        # rebuild the original display name later.

        if prevTag != tag:
            display[-1].append((tag, [token]))
            prevTag = tag
        else:
            display[-1][-1][1].append(token)
            #               ^- token
            #            ^---- tag
            #        ^-------- ingredient

            # ---- DATA ----
            # build a dict grouping tokens by their tag

            # initialize this attribute if this is the first token of its kind
        if tag not in data[-1]:
            data[-1][tag] = []

        # HACK: If this token is a unit, singularize it so Scoop accepts it.
        if tag == "unit":
            token = utils.singularize(token)

        data[-1][tag].append(token)

    # reassemble the output into a list of dicts.
    output = [
        dict([(k, utils.smartJoin(tokens)) for k, tokens in ingredient.items()])
        for ingredient in data
        if len(ingredient)
    ]

    # Add the raw ingredient phrase
    for i, v in enumerate(output):
        output[i]["input"] = utils.smartJoin(
            [" ".join(tokens) for k, tokens in display[i]])

    return output

def parse_ingredient(sent):
    """ingredient parsing logic"""
    sentence_features = get_sentence_features(sent)
    tags = tagger.tag(sentence_features)
    tagger_output = zip(sent2tokens(sentence_features), tags)
    parsed_ingredient =  format_ingredient_output(tagger_output)
    if parsed_ingredient:
        parsed_ingredient[0]['name'] = parsed_ingredient[0].get('name','').strip('.')
    return parsed_ingredient

def parse_recipe_ingredients(ingredient_list):
    """Wrapper around parse_ingredient so we can call it on an ingredient list"""
    sentences = tokenizer.tokenize(ingredient_list)
    sentences = [sent.strip('\n') for sent in sentences]
    ingredients = []
    for sent in sentences:
        ingredients.extend(parse_ingredient(sent))
    return ingredients

ingre_lists = []

for recipe in recipes:    
    ingre_str = ". ".join(recipe['ingredients'])
    parsed_ingre_list = parse_recipe_ingredients(ingre_str)
    
    ingre_store = []
    for ingre in parsed_ingre_list:
        if not "I-NAME" in ingre:
            ingre["I-NAME"] = ""
        if not "B-NAME" in ingre:
            ingre["B-NAME"] = ""
        ingre_text = ingre['B-NAME'] + " "+ ingre["I-NAME"]
        ingre_store.append(ingre_text)
        
    #print(ingre_store)
    item={"id":recipe['origin_id'], "ingredients": ingre_store}
    ingre_lists.append(item)


dim = len(ingre_lists)
ingre_dist_matrix = np.zeros((dim, dim))

from math import*
  
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

for i in range(dim):
    for j in range(i+1,dim):
        jac_sim = jaccard_similarity(ingre_lists[i]['ingredients'], ingre_lists[j]['ingredients'])
        ingre_dist_matrix[i][j]=jac_sim

mat = np.matrix(ingre_dist_matrix)
mat.dump("ingre_dist_matrix.dat")
#mat2 = numpy.load("my_matrix.dat")