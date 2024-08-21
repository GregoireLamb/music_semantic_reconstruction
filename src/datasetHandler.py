import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from mung.io import read_nodes_from_file
from sklearn import preprocessing
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

from src.config import Config


class DatasetHandler(Dataset):
    """
    Class to handle the datasets.
    It loads the data from the xml files and create a list of torch_geometric.data.Data object containing
    the ground true graphs for each score
    """

    def __init__(self, config: Config, root="", transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.config = config
        self.label_list = []
        self.label_encoder = preprocessing.LabelEncoder()
        self.dataset = self.config.__getitem__("dataset")
        self.labels_to_use = self.config.__getitem__("labels_to_use")
        self.label_transformer_dict = self.load_label_transform(self.labels_to_use)
        self.position_as_bounding_box = self.config.__getitem__("position_as_bounding_box")

        self.data_root_dict = {"muscima-pp": "./data/muscima-pp/v2.1/data/",
                               "muscima_measure_cut": "./data/muscima-pp/measure_cut/data/",
                               "doremi": "./data/DoReMi_v1/",
                               "doremi_measure_cut": "./data/DoReMi_v1/measure_cut/",
                               "musigraph": "./data/MUSIGRAPH/",
                               "musigraph_small": "./data/MUSIGRAPH_small/",
                               }

        self.xml_file_folder = {"muscima-pp": "annotations/",
                                "muscima_measure_cut": "annotations/",
                                "doremi": "Parsed_by_page_omr_xml/",
                                "doremi_measure_cut": "Parsed_by_measure_omr_xml/",
                                "musigraph": "xml/",
                                "musigraph_small": "xml/",
                                }

        try:
            self.data_root = root + self.data_root_dict[self.dataset]
        except KeyError:
            print(f"Error: '{self.dataset}' is not a supported dataset.")

        # load label classes
        self.label_list = np.array([*set(self.label_transformer_dict.values())], dtype=str)
        self.label_list = np.sort(self.label_list)
        self.label_encoder.classes_ = self.label_list

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.data_root, self.xml_file_folder[self.dataset])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.data_root,
                        f'data_processed/{self.labels_to_use}/bounding_box_{self.position_as_bounding_box}/')

    @property
    def raw_file_names(self):
        # The name of the files in the :obj:`self.raw_dir` folder that must be present in order to skip downloading.
        return os.listdir(self.raw_dir)  # For all files

    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in self.raw_file_names]

    def download(self):
        pass  # Not needed as file are stored locally

    def load_label_transform(self, labels_to_use):
        """
        Load the file containing the granularity used to one hot encode the primitive labels
        """
        if os.path.isfile(f'{self.root}./data/labels_and_links/{labels_to_use}.txt'):
            with open(f'{self.root}./data/labels_and_links/{labels_to_use}.txt', 'r') as file:
                lines = file.readlines()
            label_list = np.array([line.strip().split(',') for line in lines])
            dict = {}
            for dataset_label, label in label_list:
                dict[dataset_label] = label
            return dict
        else:
            raise FileNotFoundError(f"Label file not correct.\n"
                                    f"File {self.root}./data/labels_and_links/{labels_to_use}.txt not found")

    def process(self):
        """
        Process the dataset.
        It will encode the xml files into torch_geometric.data.Data objects and save them in the processed_dir
        """
        # Skip process if files already exist
        if len(self.raw_paths) == len(os.listdir(self.processed_dir)):
            print("All files already processed. Skipping...")
            return

        for i in (pbar := tqdm(range(len(self.raw_paths)))):
            pbar.set_description(f"Encoding XML scores into PyG dataset")

            if not osp.isfile(osp.join(self.processed_dir,
                                       f'{self.raw_file_names[i]}.pt')):  # Check if the file is already processed
                nodes_list = read_nodes_from_file(self.raw_paths[i])
                nodes_list = self.filter_and_rename_nodes_list(nodes_list)
                data = self._parse_score(nodes_list)

                torch.save(data, osp.join(self.processed_dir, f'{self.raw_file_names[i]}.pt'))

    def _parse_score(self, nodes_list: list) -> Data:
        """
        Deals with every object of the score and create a torch_geometric.data.Data object of the score
        :param nodes_list: list of object in the score (parsed with muscima.io method)
        :return: torch_geometric.data.Data object of the score
        """
        edge_index = []
        x_matrix = []
        pos = []
        # Data required ids to be continuous and to start from 0 while the xml file does not have this property
        correspondence_id_music_id = {c.id: i for i, c in enumerate(nodes_list)}

        for c in nodes_list:
            features_line = np.zeros(len(self.label_list))
            features_line[int(self.label_encoder.transform([c.class_name])[0])] = 1

            if not self.position_as_bounding_box:
                posy, posx = c.middle
                features_line = np.append(features_line, [c.height, c.width, posy, posx], axis=None)
            elif self.position_as_bounding_box:
                features_line = np.append(features_line, c.bounding_box, axis=None)

            x_matrix.append(features_line)
            pos.append([c.middle[1], c.middle[0]])
            for out in c.outlinks:
                if correspondence_id_music_id.get(out) is not None:  # Object filtering might cut some useless links
                    edge_index.append((correspondence_id_music_id[c.id], correspondence_id_music_id[out]))

        node_feats = torch.as_tensor(np.array(x_matrix), dtype=torch.float)
        node_feats = torch.as_tensor(node_feats, dtype=torch.float)

        edge_index = torch.as_tensor(edge_index, dtype=torch.int64)
        pos = torch.as_tensor(pos, dtype=torch.float)

        return Data(x=node_feats, edge_index=edge_index.t().contiguous(), pos=pos)

    def filter_and_rename_nodes_list(self, nodes_list: list):
        """
        Filter and rename the nodes_list
        :param nodes_list: list of nodes_list
        :return: list of filtered and renamed nodes_list
        """
        if self.dataset not in ['muscima-pp', 'doremi', 'musigraph', 'muscima_measure_cut', 'doremi_measure_cut']:
            print(f"Warning: filter_and_rename_nodes_list only implemented for {['muscima-pp', 'doremi', 'musigraph', 'muscima_measure_cut', 'doremi_measure_cut']}")

        nodes_list_tmp = []

        if self.label_transformer_dict.get('primitive', None) == 'primitive': # mono label (experience 1)
            for c in nodes_list:
                c.set_class_name('primitive')
                nodes_list_tmp.append(c)
            return nodes_list_tmp

        for c in nodes_list:
            if c.class_name in self.label_transformer_dict:
                c.set_class_name(self.label_transformer_dict[c.class_name])
                nodes_list_tmp.append(c)
        return nodes_list_tmp

    def get_score_image(self, score: int):
        """
        Return the image of a score for the visualisation
        """
        score_name = self.raw_file_names[score]
        img = None

        if self.dataset == 'muscima-pp' or self.dataset == 'muscima_measure_cut':
            path = os.path.abspath(self.data_root + '../images/fulls/')
            img = plt.imread(f"{path}/{score_name.rsplit('.xml', 1)[0]}.png")
            if self.dataset == 'muscima-pp':
                img = 1 - img
                img = np.stack((img,) * 3, axis=-1)
        elif self.dataset == 'doremi':
            part1 = re.search(r'_(.*?)-layout', score_name).group(1)
            page_number = re.search(r'Page_(\d+)', score_name).group(1)
            formatted_page_number = f"{int(page_number):03}"
            name_img = f"{part1}-{formatted_page_number}"
            img = plt.imread(f"{self.data_root}images/{name_img}.png")
        elif self.dataset.startswith('musigrap'):
            img = plt.imread(f"{self.data_root}images/{score_name.rsplit('.xml', 1)[0]}.png")
            print(f"Getting musigraph image: \n {self.data_root}images/{score_name.rsplit('.xml', 1)[0]}.png")
        elif self.dataset == 'doremi_measure_cut':
            img = plt.imread(f"{self.data_root}Images/{score_name.rsplit('.xml', 1)[0]}.png")
        else:
            print(f"No implementation found to display score for {self.dataset}.")

        return img

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(osp.join(self.processed_dir, f'{self.raw_file_names[idx]}.pt'))
        return data
