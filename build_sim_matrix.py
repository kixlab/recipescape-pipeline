import numpy as np
import json
import pprint as pp

from anytree import RenderTree
from zss import *

from retrieve_recipes import *
from im2recipe_w2v_weight import *

import timeit

recipes = get_all_recipes('chocochip') #list

def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.label))
        
def make_nodes(sentence):
    action = Node(label=sentence['word'], nodetype='action')
    for ing in sentence['ingredient']:
        action.addkid(Node(label=ing, nodetype='ingredient'))
    return action

ids=[]
for recipe in recipes:
    ids.append(recipe['id'])

ids = list(set(ids)) #how many recipes - determined by distict IDs


tree_list = []
for recipe_id in ids:
    recipe_inst = []
    for recipe in recipes:
        if recipe['id'] == recipe_id:
            #pp.pprint(recipe['tree'])
            recipe_inst += recipe['tree']
    
    node_list=[]

    tree1 = Node(label=recipe_inst[0]['word'], nodetype='action')
    for ing in recipe_inst[0]['ingredient']:
        tree1.addkid(Node(label=ing, nodetype='ingredient'))

    myroot = tree1
    recipe_inst = recipe_inst[1:]
    for sentence in recipe_inst:
        myroot.addkid(make_nodes(sentence), before=True)
        myroot = Node.get_children(myroot)[0]

    tree_list.append(tree1)

dim = len(tree_list)
dist_matrix = np.zeros((dim, dim))

#let's start measuring execution time
start_time = timeit.default_timer()

for i in range(dim):
    for j in range(i+1,dim):
        dist_matrix[i][j] = simple_distance(tree_list[i], tree_list[j])
        #print(i,j, dist_matrix[i][j])

elapsed = timeit.default_timer() - start_time
print(elapsed)
#end measuring execution time

mat = np.matrix(dist_matrix)
rec_ids = np.matrix(ids)
mat.dump("dist_matrix.dat")
rec_ids.dump("recipe_ids.dat")
#mat2 = numpy.load("my_matrix.dat")