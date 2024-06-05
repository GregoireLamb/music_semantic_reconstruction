import os

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.KNNGraphContainer import KNNGraphContainer
from src.config import Config
from src.datasetHandler import DatasetHandler


class Loader:
    """
    Offers instantiated KNNGraphContainer in batches for training and testing
    """

    def __init__(self, config: Config, device: torch.device, root="./") -> None:
        self.config = config
        self.device = device
        self.root = root
        self.data = []

        self.normalisation_scale = {"muscima-pp": (3403., 2354.5),
                                    "doremi": (2292.5, 1806.5),
                                    "musigraph": (2354.5, 310.0),
                                    "musigraph_small": (2354.5, 310.0),
                                    }

        self.scale = self.normalisation_scale[config.__getitem__("dataset")]

    def load(self, datasetHandler: DatasetHandler) -> None:
        """
        Load the dataset in memory and process the scores
        :param datasetHandler: dataset to load or transform in KNNGraphContainer
        :return: None
        """
        index = 0
        self.datasetHandler = datasetHandler

        dataset_name = self.config.__getitem__("dataset")
        labels_to_use = self.config.__getitem__("labels_to_use")
        position_as_bounding_box = "bounding_box" if self.config.__getitem__(
            "position_as_bounding_box") else "pos_height_width"
        n_neighbors_KNN = self.config.__getitem__("n_neighbors_KNN")
        prefilter_KNN = "prefilter" if self.config.__getitem__("prefilter_KNN") else "no_prefilter"
        normalize_positions = "normalized" if self.config.__getitem__("normalize_positions") else "not_normalized"

        file_name = (f'data_loader_{dataset_name}_{labels_to_use}_{position_as_bounding_box}_{n_neighbors_KNN}_'
                     f'{prefilter_KNN}_{normalize_positions}.pt')

        if os.path.isfile(f'{self.root}./data/loader_snapshot/{file_name}'):  # load the preprocessed data if it exists
            print("Snapchot for loader found, loading ...")
            self.data = torch.load(f'{self.root}./data/loader_snapshot/{file_name}')
            self.set_default_train_val_test_split(self.config.__getitem__("dataset"))
            return

        for score in (pbar := tqdm(range(len(self.datasetHandler)))):
            pbar.set_description(f"Processing dataset")
            data_score = KNNGraphContainer(self.datasetHandler.get(score),
                                           index,
                                           self.datasetHandler.label_encoder,
                                           scale=self.normalisation_scale[dataset_name],
                                           config=self.config,
                                           device=self.device,
                                           root=self.datasetHandler.data_root)
            self.data.append(data_score)
            index += 1

        # save self.data in a file for future use
        torch.save(self.data, f'{self.root}./data/loader_snapshot/{file_name}')
        self.set_default_train_val_test_split(dataset_name)

    def set_default_train_val_test_split(self, dataset_name: str):
        # Load a snapshot of the split if exists
        snapshot_file = f'{self.root}./data/loader_snapshot/{dataset_name}_split_train_cal_test.pt'
        if os.path.isfile(snapshot_file):
            all_split = torch.load(snapshot_file)
            self.train_scores, self.validation_scores, self.test_scores = all_split
            return

        elif dataset_name.endswith("_small"):
            self.train_scores = [0, 1, 2, 3, 4]
            self.validation_scores = [0, 1, 2, 3, 4]
            self.test_scores = [0, 1, 2, 3, 4]
            return

        else:
            split_location_dict = {"muscima-pp": self.root+"./data/muscima-pp/v2.1/specifications/",
                                   "doremi": self.root+"./data/DoReMi_v1/",
                                   "musigraph": self.root+"./data/MUSIGRAPH/"}

            split_location = split_location_dict.get(dataset_name, None)

            if split_location is None:
                raise ValueError(f"Dataset {dataset_name} has no split file location registered")

            train_ids = self.read_ids_file(f'{split_location}/train.ids')
            validation_ids = self.read_ids_file(f'{split_location}/validation.ids')
            test_ids = self.read_ids_file(f'{split_location}/test.ids')


        # elif dataset_name == "muscima_measure":
        #     train_ids = self.read_ids_file(f'{self.root}./data/MUSCIMA_measure/train_manual.ids')
        #     validation_ids = self.read_ids_file(f'{self.root}./data/MUSCIMA_measure/validation_manual.ids')
        #     test_ids = self.read_ids_file(f'{self.root}./data/MUSCIMA_measure/test_manual.ids')
        #
        #     self.train_scores = []
        #     self.validation_scores = []
        #     self.test_scores = []
        #
        #     for list, set in [(self.train_scores, train_ids),
        #                       (self.validation_scores, validation_ids),
        #                       (self.test_scores, test_ids)]:
        #         for score_name in self.datasetHandler.raw_file_names:
        #             for large_score_name in set:
        #                 if score_name.startswith(large_score_name[:-4]):  # remove the .xml
        #                     list.append(self.datasetHandler.raw_file_names.index(score_name))
        #
        #     all_split = [self.train_scores, self.validation_scores, self.test_scores]
        #     torch.save(all_split, f'{self.root}/data/loader_data/muscima_measure_split_train_val_test.pt')

        self.train_scores = [self.datasetHandler.raw_file_names.index(filename) for filename in train_ids]
        self.validation_scores = [self.datasetHandler.raw_file_names.index(filename) for filename in validation_ids]
        self.test_scores = [self.datasetHandler.raw_file_names.index(filename) for filename in test_ids]

        all_split = [self.train_scores, self.validation_scores, self.test_scores]
        torch.save(all_split, snapshot_file)

    def get_data(self, index: int):
        return self.data[index].get_data()

    def set_split(self, train, test, validation):
        self.train_scores = train
        self.test_scores = test
        self.validation_scores = validation

    def get_dataLoader(self, split="train") -> DataLoader:
        """
        Return an object of type DataLoader with the score corresponding to the desire split
        :param split: "train", "test" or "validation"
        :return: DataLoader
        """
        all_graphs = []
        if split == "train":
            score_split = self.train_scores
        elif split == "validation":
            score_split = self.validation_scores
        elif split == "test":
            score_split = self.test_scores
        else:
            raise ValueError("Split not found")

        for score in score_split:
            graph = self.get_data(score).to(self.device)
            all_graphs.append(graph)

        data_loader = DataLoader(
            all_graphs,
            batch_size=self.config.__getitem__("batch_size"),
            shuffle=True,
        )
        return data_loader

    def read_ids_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n') for line in lines]
        return lines
