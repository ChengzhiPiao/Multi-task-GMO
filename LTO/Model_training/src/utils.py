"""Data processing utilities."""

from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import json
import csv
from texttable import Texttable
from collections import Counter

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    t.add_rows(rows)
    print(t.draw())

def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['csv']
    graphs = []
    for file in get_file_paths(dir, file_format):
        with open(file, mode='r') as file:
            g = csv.reader(file)
            graphs.append(list(g))
    return graphs

def load_all_graphs(dataset_name):
    graphs = iterate_get_graphs("./json_data/"+dataset_name+"/train", "csv")
    train_num = len(graphs)
    graphs += iterate_get_graphs("./json_data/"+dataset_name+"/test", "csv")
    test_num = len(graphs) - train_num
    graphs += iterate_get_graphs("./json_data/"+dataset_name+"/val", "csv")
    val_num = test_num
    return train_num, val_num, test_num, graphs
        
def load_mcs(mcs_dict, data_location='', dataset_name='wiki2', file_name='ground_truth.txt'):
    '''
    list(tuple)
    mcs = [(id_1, id_2, mcs_value, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    mcs_value : the number of edges in MCES

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    mcs_dict[(id_1, id_2)] = ((mcs_value), best_node_mapping_list)
    '''
    
    # path = "{}json_data/{}/{}".format(data_location, dataset_name, 'graph.txt')
    # graphs = open(path,"r")
    # GID=list()
    # N=list()
    # lines = graphs.readlines()
    # for i in lines:
    #     (id,n)=i.split()
    #     GID.append(int(id))
    #     N.append(int(n))
    # graphs.close()
    
    train_num, val_num, test_num, graphs=load_all_graphs(dataset_name)
    train_num=(train_num//40)-((train_num//40)%100)
    test_num=(test_num//40)-((test_num//40)%100)
    val_num=(val_num//40)-((val_num//40)%100)
    num=0
    N=list()
    for i in range(len(graphs)):
        g=graphs[i]
        N.append(len(g))
    for tc in range(1,50):
        path = "{}json_data/{}/{}".format(data_location, dataset_name, file_name+"_"+str(tc)+".json")
        OVERLAP = open(path, 'r')
        data = json.load(OVERLAP)
        keys=list(data.keys())
        for key in range(0,len(keys),100):
            pair_=[]
            ans_=[]
            mappings_=[]
            pair__=[]
            ans__=[]
            mappings__=[]
            num+=1
            if num<=train_num or num>(train_num+val_num):
                for ii in range(10):
                    pairs=keys[key+ii]
                    (id_1,id_2)=pairs.split(',')
                    id_1=int(id_1)
                    id_2=int(id_2)
                    mcs_dict[(id_1,id_2)]=0
                continue
            for ii in range(100):
                pairs=keys[key+ii]
                (id_1,id_2)=pairs.split(',')
                id_1=int(id_1)
                id_2=int(id_2)
                match=data[pairs]
                if N[id_1]>N[id_2]:
                    (id_1,id_2)=(id_2,id_1)
                    if match!="null":
                        for i in range(len(match)):
                            (match[i][0],match[i][1])=(match[i][1],match[i][0])
                matching=list()
                for i in range(N[id_1]):
                    matching.append(N[id_2])
                ans=0
                if match!="null":
                    for i in range(len(match)):
                        matching[match[i][0]]=match[i][1]

                    g1=graphs[id_1]
                    g2=graphs[id_2]
                    n1=len(g1[0])
                    n2=len(g2[0])
                    col_g1=list()
                    for i in range(n1):
                        col=""
                        for j in range(len(match)):
                            col=col+" "+g1[match[j][0]][i]
                        col_g1.append(col)
                    col_g2=list()
                    for i in range(n2):
                        col=""
                        for j in range(len(match)):
                            col=col+" "+g2[match[j][1]][i]
                        col_g2.append(col)
                    counter1 = Counter(col_g1)
                    counter2 = Counter(col_g2)
                    ans = sum((counter1 & counter2).values())*len(match)
                
                mappings=list()
                mappings.append(matching)
                if ans!=0:
                    pair_.append((id_1,id_2))
                    ans_.append(ans)
                    mappings_.append(mappings)
                else:
                    pair__.append((id_1,id_2))
                    ans__.append(ans)
                    mappings__.append(mappings)
            for ii in range(min(10,len(pair_))):
                mcs_dict[pair_[ii]]=((ans_[ii],0.0,0.0,0.0),mappings_[ii])
            for ii in range(max(0,10-len(pair_))):
                mcs_dict[pair__[ii]]=((ans__[ii],0.0,0.0,0.0),mappings__[ii])
        OVERLAP.close()