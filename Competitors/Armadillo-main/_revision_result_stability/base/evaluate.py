from _revision_result_stability.base.armadillo import *
#________________________________________________With correct label
def model_inference(model: Armadillo, data_loader: DataLoader, device: str) -> set:
    model.eval()

    y_pred = None
    y_true = None
    with torch.no_grad():
        for batch in data_loader:
            # to device
            emb_l = model(batch[0].to(device))
            emb_r = model(batch[1].to(device))

            logits = F.cosine_similarity(emb_l, emb_r, dim=1)

            logits[logits < 0] = 0

            y = batch[2]
                # Save the predictions and the labels
            if y_pred is None:
                y_pred = logits  
                y_true = batch[2]
            else:
                y_pred = torch.cat((y_pred, logits))
                y_true = torch.cat((y_true, y))

    return y_pred, y_true

def test(model: Armadillo, test_dataset: GraphTriplesDataset, batch_size: int=64, 
         num_workers: int=0) -> torch.Tensor:
    eval_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,  
                             num_workers=num_workers, 
                             shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred, y_true = model_inference(
        model,
        eval_loader,
        device
    )
    return compute_metrics(y_pred, y_true)

def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """Function to compute the necessary evaluation metrics

    Args:
        y_pred (torch.Tensor): prediction tensor
        y_true (torch.Tensor): true label tensor

    Returns:
        dict: contains the following metrics: mse, mae, variance, min_ae, abs_diff_tensor 
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mse = F.mse_loss(y_pred, y_true.to(device))
        abs_diff_tensor = torch.abs(y_pred - y_true.to(device))
        mae = torch.mean(abs_diff_tensor)
        variance = torch.var(abs_diff_tensor)
        min_ae = torch.min(abs_diff_tensor)
        max_ae = torch.max(abs_diff_tensor)
    except:
        mse = None
        abs_diff_tensor = None
        mae = None
        variance = None
        min_ae = None
        max_ae = None
    return {'mse':mse, 'mae':mae, 'variance':variance, 'min_ae':min_ae, 'max_ae':max_ae, 'abs_diff_tensor':abs_diff_tensor, 'predictions':y_pred.cpu(), 'overlaps_true':y_true.cpu(), 'abs_diff_tensor':abs_diff_tensor.cpu()}


def re_evaluate(graphs_path,triple_file,out,model_file):
    print('loading dataset')
    test_triples = pd.read_csv(triple_file)[['r_id','s_id','a%']]
    print('loading graphs')
    with open(graphs_path,'rb') as f:
        graphs = pickle.load(f) 
    print('building graph dataset')
    test_dataset = GraphTriplesDataset(test_triples, graphs)
    device = torch.device("cuda" 
    if torch.cuda.is_available() else "cpu")
    print('creating model')
    model = Armadillo(model_file=model_file)
    print('computing overlaps')
    execution_insights_test = test(model, test_dataset, batch_size=64)
    print('saving results')
    with open(out, 'wb') as f:
        pickle.dump(execution_insights_test, f)
root = ''
armadillo_w_w = {
    'graphs_path':root+'/WikiTables/dictionaries/graph_dict.pkl',
    'triple_file':root+'/WikiTables/test.csv',
    'out':root+'/WikiTables/evaluation/re_eval_armadillo_w_w.pkl',
    'model_file':root+'/WikiTables/models/armadillo_wiki.pth'
}
armadillo_g_w = {
    'graphs_path':root+'/WikiTables/dictionaries/graph_dict.pkl',
    'triple_file':root+'/WikiTables/test.csv',
    'out':root+'/WikiTables/evaluation/re_eval_armadillo_g_w.pkl',
    'model_file':root+'/GitTables/models/armadillo_git.pth'
}
armadillo_g_g = {
    'graphs_path':root+'/GitTables/dictionaries/graph_dict.pkl',
    'triple_file':root+'/GitTables/test.csv',
    'out':root+'/GitTables/evaluation/re_eval_armadillo_g_g.pkl',
    'model_file':root+'/GitTables/models/armadillo_git.pth'
}
armadillo_w_g = {
    'graphs_path':root+'/GitTables/dictionaries/graph_dict.pkl',
    'triple_file':root+'/GitTables/test.csv',
    'out':root+'/GitTables/evaluation/re_eval_armadillo_w_g.pkl',
    'model_file':root+'/WikiTables/models/armadillo_wiki.pth'
}
armadillo_g_table_querying = {
    'graphs_path':root+'/GitTables/table_querying/dictionaries/graph_dict_table_querying.pkl',
    'triple_file':root+'/GitTables/table_querying/table_querying_stats.csv',
    'out':root+'/GitTables/table_querying/dictionaries/embedding_dictionaries/re_eval_table_querying_armadillo_g.pkl',
    'model_file':root+'/GitTables/models/armadillo_git.pth'
}
print('armadillo_w_w')
re_evaluate(**armadillo_w_w)
print('armadillo_g_w')
re_evaluate(**armadillo_g_w)
print('armadillo_g_g')
re_evaluate(**armadillo_g_g)
print('armadillo_w_g')
re_evaluate(**armadillo_w_g)
re_evaluate(**armadillo_g_table_querying)