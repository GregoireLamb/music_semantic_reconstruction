import copy
from datetime import datetime

import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.optim import lr_scheduler
from tqdm import tqdm

from src.dataLoader import Loader
from src.datasetHandler import DatasetHandler
from src.utils import *


def train(config: Config, writer: SummaryWriter, loader: Loader, device=torch.device('cpu'),
          model_to_load=False) -> Model:
    """
    Training loop
    """
    # INIT
    best_validation_loss, start_epoch, best_epoch = np.inf, 0, -1

    init_n_label = len(loader.datasetHandler.label_list) + 4  # +4 is for the position
    model = Model(init_n_label, config).to(device)
    best_model = Model(init_n_label, config).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.__getitem__("learning_rate"))
    best_optimizer = torch.optim.Adam(list(model.parameters()), lr=config.__getitem__("learning_rate"))

    if model_to_load != False:
        model, best_model, optimizer, starting_epoch, best_validation_loss = load_model(model_to_load,
                                                                                        model, best_model,
                                                                                        optimizer, config)
    if config.__getitem__("start_from_best_model"):
        best_optimizer = copy.deepcopy(optimizer)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=config.__getitem__("scheduler_factor"),
                                               patience=config.__getitem__("scheduler_patience"))
    config.__setitem__('model_config', model.__repr__())
    model.train().to(device)

    data_loader = loader.get_dataLoader(split="train")

    # Setup weight and loss function
    weight = config.__getitem__("weight_imbalance")
    if weight == -1:
        weight = compute_weight_imbalance(data_loader)
        config.config["weight_imbalance"] = weight

    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device)).to(device)

    for epoch in (pbar := tqdm(range(start_epoch, start_epoch + config.__getitem__("n_epochs")))):
        pbar.set_description(f"Training")
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        loss_values = []
        for graph_batch in data_loader:
            # graph_batch = graph_batch.to(device)
            optimizer.zero_grad()

            if config.__getitem__("undirected_edges"):
                graph_batch = T.Compose([T.ToUndirected()])(graph_batch)

            # if config.__getitem__("use_sparse_adj"):
            #     graph_batch = T.Compose([T.ToSparseTensor()])(graph_batch)
            #     # Only edge_index is turned into adj_t (the other tensor arn't but having them sparse leads to error)
            #     predictions = model(x=graph_batch.x, pos=graph_batch.pos, edge_index=graph_batch.adj_t).to(device)
            # elif config.__getitem__("hetero_data"):
            #     predictions = model(x=graph_batch.x_dict, pos=graph_batch.pos,
            #                         edge_index=graph_batch.edge_index_dict).to(device)
            # else:
            predictions = model(x=graph_batch.x, pos=graph_batch.pos, edge_index=graph_batch.edge_index).to(device)
            truth = graph_batch.truth.to(device)
            loss = loss_function(predictions, truth)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

        writer.add_scalar("Loss/train", np.mean(loss_values), epoch)
        writer.flush()

        validation_loss = validate(loader, model, device, loss_function)
        writer.add_scalar("Loss/validation", validation_loss, epoch)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer)
            best_epoch = epoch

        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_loss)

        # Jump back to best known model if lr change
        if config.__getitem__("jump_back_on_lr_change"):
            if previous_lr != optimizer.param_groups[0]["lr"]:
                model = copy.deepcopy(best_model)
                optimizer = torch.optim.Adam(list(model.parameters()), lr=optimizer.param_groups[0]["lr"])
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=config.__getitem__("scheduler_factor"),
                                                           patience=config.__getitem__("scheduler_patience"))

    writer.add_text("config", config.__repr__())
    writer.flush()

    result = {
        "best_model": best_model,
        "best_optimizer": best_optimizer,
        "model": model,
        "optimizer": optimizer,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "start_epoch": start_epoch,
        "final_epoch": start_epoch + config.__getitem__("n_epochs")
    }

    return result


def validate(loader: Loader, model: Model, device: torch.device, loss_function: torch.nn.Module):
    loss_values = []
    data_loader = loader.get_dataLoader(split="validation")
    for graph_validation_batch in data_loader:
        if config.__getitem__("undirected_edges"):
            graph_validation_batch = T.Compose([T.ToUndirected()])(graph_validation_batch)
        predictions = model(x=graph_validation_batch.x, pos=graph_validation_batch.pos,edge_index=graph_validation_batch.edge_index)
        predictions = predictions.view(-1).to(device)
        loss_val = loss_function(predictions, graph_validation_batch.truth)
        loss_values.append(loss_val.item())
    return np.mean(loss_values)


def test(model: Model):
    model = model.to(device)
    model.eval()
    link_analyse_dict = {}
    confusion_mat = [[0, 0], [0, 0]]
    accuracy, edit_distances, music_error_rate = [], [], []

    score_loader = loader.get_dataLoader(split="test")
    do_visualize_first_score = config.__getitem__("visualize_first_score")

    for graph in (pbar := tqdm(score_loader)):
        pbar.set_description(f"Testing ")

        if config.__getitem__("undirected_edges"):
            graph = T.Compose([T.ToUndirected()])(graph)

        predictions = model(x=graph.x, pos=graph.pos, edge_index=graph.edge_index)
        predictions = predictions.view(-1)
        truth = np.round(graph.truth.tolist())
        predictions = np.round(predictions.tolist())

        ged, mer = compute_ged_mer_for_batch(graph, predictions)
        edit_distances.extend(ged)
        music_error_rate.extend(mer)

        link_analyse_dict = perform_predictions_analysis(graph, predictions, truth,
                                                         loader.datasetHandler.label_encoder,
                                                         dict=link_analyse_dict)

        if do_visualize_first_score:
            first_score = graph.to_data_list()[0]
            score_image = loader.datasetHandler.get_score_image(first_score.index)
            predictions_first_score = predictions[0:first_score.edge_index.shape[1]]
            generate_visualizations(first_score, predictions_first_score, writer, score_image,
                                    loader=loader, dataset=config.__getitem__('dataset'))
            do_visualize_first_score = False

        conf = confusion_matrix(truth, predictions)

        if len(conf) == 1:  # case there is only positive or negative edges
            print("Only positive or negative edges in the confusion matrix ", len(conf))
            if truth[0] == 1:
                conf = np.array([[0, 0], conf[0]])
            else:
                conf = np.array([conf[0], [0, 0]])

        confusion_mat += conf
        accuracy.append(accuracy_score(truth, predictions))
    save_confusion_matrix(confusion_mat=confusion_mat, writer=writer)
    writer.add_scalar(tag="Accuracy/test", scalar_value=np.mean(accuracy),
                      global_step=int(datetime.now().timestamp()))
    writer.add_scalar(tag="Mean_GED/test",
                      scalar_value=np.mean(edit_distances),
                      global_step=int(datetime.now().timestamp()))
    writer.add_scalar(tag="Mean_MER/test",
                      scalar_value=np.mean(music_error_rate),
                      global_step=int(datetime.now().timestamp()))

    writer.add_histogram(tag="GED_hist/test",
                         values=np.array(edit_distances))
    writer.add_histogram(tag="MER_hist/test",
                         values=np.array(music_error_rate))

    writer.flush()

    for key, value in link_analyse_dict.items():
        link_analyse_dict[key] = [np.round(value[1] / (value[0] + value[1]), decimals=5), value[0] + value[1]]
    link_analyse_dict = dict(sorted(link_analyse_dict.items(), key=lambda item: sum(item[1]), reverse=True))
    writer.add_text("Link Analysis", str(link_analyse_dict))
    writer.flush()


def save_model(config: Config, train_results: dict, run_name: str):
    if config.__getitem__("save_model"):
        print("saving model ...")
        torch.save(train_results, './models/' + run_name + '.pth')


if __name__ == '__main__':
    config = Config()
    seed_everything_(config.__getitem__("seed"))
    run_name = 'run_' + datetime.now().strftime('%m-%d-%Y_%H-%M')
    writer = SummaryWriter('./runs/'+run_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = Loader(config, device)

    dataset = DatasetHandler(config)
    loader.load(dataset)

    train_results = train(config=config,
                          writer=writer,
                          loader=loader,
                          device=device,
                          model_to_load=config.__getitem__("load_model"))

    print("Best model is reached at epoch: ", train_results["best_epoch"])

    test(train_results["best_model"])

    save_model(config, train_results, run_name)

    writer.flush()
    writer.close()
