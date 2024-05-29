import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from muscima.io import parse_cropobject_list
from sklearn import preprocessing
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

from src.config import Config


class DatasetHandler(Dataset):
    """
    Class to handle the dataset. It will load the data from the xml files and create a list of torch_geometric.data.Data object
    """

    def __init__(self, config: Config, root="", transform=None, pre_transform=None, pre_filter=None):
        self.config = config
        self.label_encoder = preprocessing.LabelEncoder()
        self.dataset = self.config.__getitem__("dataset")
        self.labels = self.config.__getitem__("labels")
        self.position_as_bounding_box = self.config.__getitem__("position_as_bounding_box")

        self.data_root_dict = {"muscima++": "./data/muscima++/v1.0/data/",
                               # "musigraph": "./data/MUSIGRAPH/",
                               # "muscima_measure": "./data/MUSCIMA_measure/",
                               # "DOREMI": "./data/DoReMi_v1/",
                               # "muscima++_small": "./data/SMALL_MUSCIMA/",
                               # "musigraph_small": "./data/SMALL_MUSIGRAPH/",
                               # "muscima_measure_small": "./data/MUSCIMA_measure/",
                               # "DOREMI_small": "./data/SMALL_DOREMI/"
                               }

        self.xml_file_folder = {"muscima++": "cropobjects_manual/",
                                # "musigraph": "xml/",
                                # "muscima_measure": "xml/",
                                # "DOREMI": "Parsed_by_page_omr_xml/",
                                # "muscima++_small": "xml/",
                                # "musigraph_small": "xml/",
                                # "muscima_measure_small": "xml/",
                                # "DOREMI_small": "Parsed_by_page_omr_xml/"
                                }

        try:
            self.data_root = root + self.data_root_dict[self.dataset]
        except KeyError:
            print(f"Error: '{self.dataset}' is not a supported dataset.")

        # self.raw_dir = osp.join(self.data_root, xml_file_folder[self.dataset])
        # self.scores_names = os.listdir(self.raw_dir)

        # load label classes
        with open(f'./data/labels/{self.labels}.txt', 'r') as file:
            lines = file.readlines()
        self.label_list = np.array([line.strip() for line in lines])  # TODO -> need to remove the "'" ?
        self.label_encoder.classes_ = self.label_list

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.data_root, self.xml_file_folder[self.dataset])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.data_root, f'data_processed/{self.labels}/bounding_box_{self.position_as_bounding_box}/')

    @property
    def raw_file_names(self):
        # The name of the files in the :obj:`self.raw_dir` folder that must be present in order to skip downloading.
        return os.listdir(self.raw_dir)  # For all files

    @property
    def processed_file_names(self):
        return [f'{i}.pt' for i in self.raw_file_names]

    def download(self):
        pass  # Not needed as file are stored locally

    def process(self):
        """
        Process the dataset. It will encode the xml files into torch_geometric.data.Data objects and save them in the processed_dir
        """
        # Skip process if files already exist
        if len(self.raw_paths) == len(os.listdir(self.processed_dir)):
            print("All files already processed. Skipping...")
            return

        for i in (pbar := tqdm(range(len(self.raw_paths)))):
            pbar.set_description(f"Encoding XML scores into PyG dataset")

            if not osp.isfile(osp.join(self.processed_dir,
                                       f'{self.raw_file_names[i]}.pt')):  # Check if the file is already processed
                cropobjects = parse_cropobject_list(self.raw_paths[i])
                cropobjects = self.filter_and_rename_cropobjects(cropobjects)
                data = self._parse_score(cropobjects)

                torch.save(data, osp.join(self.processed_dir, f'{self.raw_file_names[i]}.pt'))

    def _parse_score(self, cropobjects: list) -> Data:
        """
        Deals with every object of the score and create a torch_geometric.data.Data object of the score
        :param cropobjects: list of object in the score (parsed with muscima.io method)
        :return: torch_geometric.data.Data object of the score
        """
        edge_index = []
        x_matrix = []
        pos = []
        # Data required ids to be continuous and to start from 0 while the xml file does not have this property
        correspondence_id_music_id = {c.objid: i for i, c in enumerate(cropobjects)}

        for c in cropobjects:
            features_line = np.zeros(len(self.label_list))
            features_line[int(self.label_encoder.transform([c.clsname])[0])] = 1

            if not self.position_as_bounding_box:
                posy, posx = c.middle
                features_line = np.append(features_line, [c.height, c.width, posy, posx], axis=None)
            elif self.position_as_bounding_box:
                features_line = np.append(features_line, c.bounding_box, axis=None)

            x_matrix.append(features_line)
            pos.append([c.middle[1], c.middle[0]])  # TODO assert order (fine)
            for out in c.outlinks:
                if correspondence_id_music_id.get(out) is not None:  # Object filtering might cut some useless links
                    edge_index.append((correspondence_id_music_id[c.objid], correspondence_id_music_id[out]))

        node_feats = torch.as_tensor(np.array(x_matrix), dtype=torch.float)
        node_feats = torch.as_tensor(node_feats, dtype=torch.float)

        edge_index = torch.as_tensor(edge_index, dtype=torch.int64)
        pos = torch.as_tensor(pos, dtype=torch.float)

        return Data(x=node_feats, edge_index=edge_index.t().contiguous(), pos=pos)

    def filter_and_rename_cropobjects(self, cropobjects: list):
        """
        Filter and rename the cropobjects
        :param cropobjects: list of cropobjects
        :return: list of filtered and renamed cropobjects
        """
        # if self.dataset == 'DOREMI':
        #     dict_doremi2std = {'noteheadBlack': 'notehead-full',
        #                        'noteheadHalf': 'notehead-empty',
        #                        'accidentalNatural': 'natural',
        #                        'accidentalFlat': 'flat,',
        #                        'accidentalSharp': 'sharp',
        #                        'beam': 'beam',
        #                        'stem': 'stem',
        #                        'flag16thDown': '16th_flag',
        #                        'flag16thUp': '16th_flag',
        #                        'flag8hDown': '8th_flag',
        #                        'flag8thUp': '8th_flag',
        #                        }
        #     # Filter and rename the classes
        #     cropobjects_tmp = []
        #     for c in cropobjects:
        #         if c.clsname in dict_doremi2std:
        #             c.clsname = dict_doremi2std[c.clsname]
        #             cropobjects_tmp.append(c)
        #     cropobjects = cropobjects_tmp

        if self.dataset == 'muscima++':
            cropobjects = [c for c in cropobjects if c.clsname in self.label_list]
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented yet")
        return cropobjects

    def get_score_image(self, score: int):
        # TODO simplify and handle other datasets
        name = self.scores_names[score]
        if self.dataset == 'muscima++':
            img = plt.imread(f"{self.data_root}data/images/fulls/{name.rsplit('.xml', 1)[0]}.png")
        elif self.dataset == 'DOREMI':
            # Extracting the part after the first underscore and before '-layout'
            part1 = re.search(r'_(.*?)-layout', name).group(1)

            # Extracting the page number after 'Page_' and formatting it to three digits
            page_number = re.search(r'Page_(\d+)', name).group(1)
            formatted_page_number = f"{int(page_number):03}"

            # Combining the extracted parts into the desired format
            name_img = f"{part1}-{formatted_page_number}"
            img = plt.imread(f"{self.data_root}/images/{name_img}.png")
        else:
            img = plt.imread(f"{self.data_root}/images/{name.rsplit('.xml', 1)[0]}.png")
        return img

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(osp.join(self.processed_dir, f'{self.raw_file_names[idx]}.pt'))
        return data
