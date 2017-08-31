# recipe.hyeungshikjung.com/recipe/trees/potatosalad
# ~/recipe/recipe/<id> : get recipe using id
# ~/recipe/recipes/<dishname>: get all the recipes of dish (currently only ‘potatosalad’ is available)
# ~/recipe/clusters/<dishname>: get all clustering result of dish

import json
import requests
import pprint as pp

servername = "http://recipe.hyeungshikjung.com/"

# dishname = "potatosalad"
# recipeid = ""

def get_recipes_json(dishname):
    #getting annotated text instructions for all recipes for dishname

    try:
        r = requests.get(servername+"recipe/recipes/"+dishname)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        sys.exit(1)

    #recipes = r.json()
    #pp.pprint(recipes) 
    return r.json()


def get_all_recipes(dishname):
    #getting annotated text instructions for all recipes for dishname

    try:
        r = requests.get(servername+"recipe/trees/"+dishname)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        sys.exit(1)

    #recipes = r.json()
    #pp.pprint(recipes) 
    return r.json()
    
def get_single_recipe(recipeid):
    #getting a single recipe 
    
    try:
        r = requests.get(servername+"recipe/recipe/"+recipeid)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        sys.exit(1)

    return r.json()

def get_clusters(dishname):
    try:
        r = requests.get(servername+"recipe/clusters"+dishname)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        sys.exit(1)

    return r.json()