from Code._training_pipeline import *
import pickle
from tqdm import tqdm

def model_inference_rep(model: Armadillo, data_loader: DataLoader, device: str) -> set:
    """Genertion of 2 sets of labels (predicted and real)

    Args:
        model (Armadillo): the model to test
        data_loader (DataLoader): dataloader containing the test data
        device (str): the device the model is working with

    Returns:
        set: a set of 2 tensors (PREDICTIONS, LABELS)
    """
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

def test_rep(model: Armadillo, test_dataset: GraphTriplesDataset, batch_size: int=64, 
         num_workers: int=0) -> torch.Tensor:
    """Perform the testing operation

    Args:
        model (Armadillo): the model to test
        test_dataset (GraphTriplesDataset): testing dataset generated with train_test_valid_split
        batch_size (int, optional): size of the batches. Defaults to 64.
        num_workers (int, optional): NA. Defaults to 0.

    Returns:
        torch.Tensor: torch.Tensor: the average loss on the test set 
    """
    eval_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,  
                             num_workers=num_workers, 
                             shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    # to device

    y_pred, y_true = model_inference_rep(
        model,
        eval_loader,
        device
    )
    return compute_metrics(y_pred, y_true)

if __name__ == '__main__':
    root = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/'
    model_file = root+'/models/armadillo_wiki.pth'
    test_triple_file_path = root+'/test.csv'
    graph_dict_path = root + '/dictionaries/graph_dict.pkl'
    print('Opening graph dict')
    with open(graph_dict_path, 'rb') as f:
        gd = pickle.load(f)
    print('Loading triple file')
    ttf = pd.read_csv(test_triple_file_path)
    print('Loading model')
    model = Armadillo(model_file=model_file)
    print('Creating dataset')
    test_dataset = GraphTriplesDataset(ttf, gd)
    print('Starting evaluation')
    execution_insights_test = test_rep(model, test_dataset, 1)
    print(f'MAE: {execution_insights_test["mae"]}')