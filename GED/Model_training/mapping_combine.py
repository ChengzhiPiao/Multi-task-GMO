import torch
import random
import pickle
import json

f = open("App_Bmao_matching_IMDBMulti.txt", "r")

all = [list(temp.strip().split()) for temp in f.readlines()]

f.close()

val_dataset = torch.load('./val_dataset_GED_IMDBMulti_70.pt')
n=len(val_dataset['mcs'])
li=[[i for i in range(300)] for _ in range(900)]

f = open("ged_gr", "r")
gr_all = [list(map(int,temp.strip().split())) for temp in f.readlines()]
f.close()

test_case=dict()
num=0
for i in range(0,900):
    for j in range(1200,1500):
        if gr_all[i][j]>=0:
            test_case[(i,j-1200)]=num
            num+=1
    for j in range(1200,1500):
        if gr_all[i][j]<0:
            test_case[(i,j-1200)]=num
            num+=1


# f=open("ground_truth_GED_IMDBMulti_70.txt","w")
# f.write(str(900*300)+"\n")
# # random.seed(1)
# # for i in range(0,900):
# #     random.shuffle(li[i])
# for i in range(0,900):
#     for j in range(0,300):
#         f.write(str(int(val_dataset['mcs'][test_case[(i,li[i][j])]]))+"\n")
# f.close()
with open("./json_data/IMDBMulti/graph.pkl", "rb") as f:
    graphs = pickle.load(f)
f = open("val_data_GED_mapping_IMDBMulti_70.txt", "w")
# f2 = open("val_data_GED_IMDBMulti_graph_id.txt", "w")
f.write(str(900*300)+"\n")
# f2.write(str(900*300)+"\n")
for i in range(0,900):
    for jj in range(0,300):
        f.write(str(val_dataset['soft_matrix'][test_case[(i,li[i][jj])]].size(0))+" "+str(val_dataset['soft_matrix'][test_case[(i,li[i][jj])]].size(1))+"\n")
        for j in range(0,val_dataset['soft_matrix'][test_case[(i,li[i][jj])]].size(0)):
            Mapping_matrix=val_dataset['soft_matrix'][test_case[(i,li[i][jj])]][j].detach().numpy()
            for k in range(0,val_dataset['soft_matrix'][test_case[(i,li[i][jj])]].size(1)):
                f.write(str(Mapping_matrix[k])+" ")
            f.write("\n")
        for j in range(0,val_dataset['soft_matrix'][test_case[(i,li[i][jj])]].size(0)):
            mapping_matrix=val_dataset['mapping'][test_case[(i,li[i][jj])]][j].detach().numpy()
            for k in range(0,val_dataset['soft_matrix'][test_case[(i,li[i][jj])]].size(1)):
                f.write(str(int(mapping_matrix[k]))+" ")
            f.write("\n")
        graphAid = val_dataset["graphAid"][test_case[(i,li[i][jj])]]
        graphA = graphs["graph"][int(graphAid)]
        graphBid = val_dataset["graphBid"][test_case[(i,li[i][jj])]]
        graphB = graphs["graph"][int(graphBid)]
        # f2.write(str(int(graphAid))+" "+str(int(graphBid))+"\n")
        for j in range(0,graphA['n']):
            # f.write(str(graphA['labels'][j])+" ")
            f.write(str(1)+" ")
        f.write("\n")
        f.write(str(graphA['m'])+"\n")
        for j in range(0,graphA['m']):
            u = graphA["edge_index"][0, j].item() 
            v = graphA["edge_index"][1, j].item() 
            f.write(f"{u} {v}\n")
        for j in range(0,graphB['n']):
            # f.write(str(graphB['labels'][j])+" ")
            f.write(str(1)+" ")
        f.write("\n")
        f.write(str(graphB['m'])+"\n")
        for j in range(0,graphB['m']):
            u = graphB["edge_index"][0, j].item() 
            v = graphB["edge_index"][1, j].item() 
            f.write(f"{u} {v}\n")
        # for j in range(0,len(all[(i*300+j)*2+1])):
        for j in all[(i*300+jj)*2+1]:
            f.write(str(j)+" ")
        f.write("\n")
f.close()