import torch
val_dataset = torch.load('./val_dataset_MCIS_BI.pt')
import json
# f=open("ground_truth_MCIS_BI.txt","w")
#400*20=8000
# f.write("351\n")
# for i in range(0,351):
#     f.write(str(val_dataset['mcs'][i])+"\n")
# f.close()
f = open("val_data_MCIS_BI.txt", "w")
f.write("351\n")
for i in range(0,351):
    f.write(str(val_dataset['soft_matrix'][i].size(0))+" "+str(val_dataset['soft_matrix'][i].size(1))+"\n")
    for j in range(0,val_dataset['soft_matrix'][i].size(0)):
        Mapping_matrix=val_dataset['soft_matrix'][i][j].detach().numpy()
        for k in range(0,val_dataset['soft_matrix'][i].size(1)):
            f.write(str(Mapping_matrix[k])+" ")
        f.write("\n")
    # for j in range(0,val_dataset['soft_matrix'][i].size(0)):
    #     mapping_matrix=val_dataset['mapping'][i][j].detach().numpy()
    #     for k in range(0,val_dataset['soft_matrix'][i].size(1)):
    #         f.write(str(int(mapping_matrix[k]))+" ")
    #     f.write("\n")
    graphAid = val_dataset["graphAid"][i]
    graphA = json.load(open(f'./json_data/BI/graphs/{graphAid}.json', 'r'))
    graphBid = val_dataset["graphBid"][i]
    graphB = json.load(open(f'./json_data/BI/graphs/{graphBid}.json', 'r'))
    for j in range(0,graphA['n']):
        # f.write(str(graphA['labels'][j])+" ")
        f.write(str(0)+" ")
    f.write("\n")
    f.write(str(graphA['m'])+"\n")
    for j in range(0,graphA['m']):
        f.write(str(graphA['graph'][j][0])+" "+str(graphA['graph'][j][1])+"\n")
    for j in range(0,graphB['n']):
        # f.write(str(graphB['labels'][j])+" ")
        f.write(str(0)+" ")
    f.write("\n")
    f.write(str(graphB['m'])+"\n")
    for j in range(0,graphB['m']):
        f.write(str(graphB['graph'][j][0])+" "+str(graphB['graph'][j][1])+"\n")
f.close()