import copy
import os
import xml.etree.ElementTree as ET
from random import random
import re

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mung.io import read_nodes_from_file
from tqdm import tqdm


def visualize_score_with_obj(
        df: pd.DataFrame,
        score_path,
        output_path,
        dpi=300,
        measure_obj=False):
    """
    Generate an image with every bouding box and their ids on top of it from a dataframe, if measure_obj is a number,
    colors are applyed depending on the value in the column 'measure'
    :param df:
    :param score_path:
    :param output_path:
    :param dpi:
    :param measure_obj:
    :return:
    """
    color_dict = {}
    if measure_obj != False:
        for i in range(measure_obj):
            color_dict[i] = (int(random() * 255), int(random() * 255), int(random() * 255))

    # Read the image
    if DATASET == 'muscima-pp':
        image = cv2.imread(score_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.bitwise_not(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif DATASET == 'doremi':
        image = cv2.imread(score_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {score_path}")

    # Draw bounding boxes and labels_and_links on the image
    for index, row in df.iterrows():
        top, bottom, left, right = int(row['top']), int(row['bottom']), int(row['left']), int(row['right'])
        # Draw rectangle
        if measure_obj != False:
            cv2.rectangle(image, (left, top), (right, bottom), color_dict[row['measure']], 2)
        else:
            rnd_color = (int(random() * 255), int(random() * 255), int(random() * 255))
            cv2.rectangle(image, (left, top), (right, bottom), (rnd_color), 2)
        # Put class label
        cv2.putText(image, row['id'], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Display the image using Matplotlib
    plt.figure(figsize=(12, 12))
    plt.title(score_path.split('/')[-1][:-4])
    plt.imshow(image)
    plt.axis('off')

    # Save the image with increased DPI
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    # plt.show()
    plt.close()

def adapt_position(measures_image_limit, child):
    """
    Adapt the position of the Objects to the new image (subtract the min_x and min_y from the top and left values)
    """
    child = copy.deepcopy(child)
    min_x, max_x, min_y, max_y = measures_image_limit
    for subchild in child:
        if subchild.tag == 'Top':
            subchild.text = str(int(subchild.text) - min_y)
        elif subchild.tag == 'Left':
            subchild.text = str(int(subchild.text) - min_x)
    return child


def cut_out_in_links(child, id_obj_in_measure, banned_id_list):
    """
    Cut the inlnks and outlinks to objects not in the measure
    Here for object not in the measure but link to an object in the measure -(typically harpins)'
    """
    for subchild in child:
        if subchild.tag in ['Outlinks', 'Inlinks']:
            list_ids = subchild.text.split()
            ids_to_keep = list_ids.copy()
            for id in list_ids:
                if (id in banned_id_list) or (id not in id_obj_in_measure):
                    ids_to_keep.remove(id)
            subchild.text = ' '.join(ids_to_keep)
    return child


def filter_elements(element, id_relative2measure, id_obj_in_measure, measures_image_limit, banned_id_list):
    """
    Recursively filter elements of a xml file based on the id_list and adapt the obj positions.
    """
    # Create a list to hold the filtered children
    filtered_children = []

    for child in element:
        if child.tag == 'Node':
            # Find the 'Id' element within this node
            for subchild in child:
                if subchild.tag == 'Id' and subchild.text in id_relative2measure:
                    child = adapt_position(measures_image_limit, child)  # TODO WARINING: MASK NOT CORRECT ANYMORE
                    if subchild.text not in id_obj_in_measure:
                        child = cut_out_in_links(child, id_obj_in_measure, banned_id_list)
                    filtered_children.append(child)
                    break
        else:
            # Otherwise, recursively check the child's children
            filtered_child = filter_elements(child, id_relative2measure, id_obj_in_measure, measures_image_limit, banned_id_list)
            if filtered_child is not None:
                filtered_children.append(filtered_child)

    # If there are filtered children, create a new element with these children
    if filtered_children:
        new_element = ET.Element(element.tag, element.attrib)
        new_element.extend(filtered_children)
        return new_element

    # If no children were added, return None
    return None


def filter_xml(input_file, output_file, id_relative2measure, id_obj_in_measure, measures_image_limit, banned_id_list):
    """
    Generate and save the xml file with the objects corresponding to the id given
    :param input_file:
    :param output_file:
    :param id_relative2measure:
    :param id_obj_in_measure:
    :param measures_image_limit:
    :return:
    """
    # Parse the input XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Filter the root element
    new_root = filter_elements(root, id_relative2measure, id_obj_in_measure, measures_image_limit, banned_id_list)

    # Create a new tree with the filtered elements if any
    if new_root is not None:
        new_tree = ET.ElementTree(new_root)
        # Write the new XML tree to the output file
        with open(os.path.abspath(output_file), 'wb') as f:
            new_tree.write(f)


def remove_staff_lines(musical_primitives):
    musical_primitives_tmp = musical_primitives.copy()
    banned_ids = []
    for c in musical_primitives:
        if c.class_name in ['staffSpace', 'staffLine', 'staff', 'staffGrouping', 'kStaffLine']:
            musical_primitives_tmp.remove(c)
            banned_ids.append(str(c.id))
    return musical_primitives_tmp, banned_ids


def cut_measures(img_score_path: str, xml_path: str, output_path: str,
                 dpi=300):
    """
    read an xml file, determine the measure and the object belonging to it, cut the image and
    save it, rewrite the XML and modify the obj position to the cutted image

    step:
    0 - cut the staff related object
    1 - find thin bars
    2 - find measures
    3 - associate obj to measures
    4 - cut image
    5 - associate obj to measures and write xml
    6 - generate colorful full image with all obj

    """
    # 1- read XML and find the barlines
    score_name = xml_path.split("/")[-1].split(".")[0]
    # print('Processing: ', score_name)
    musical_primitives = read_nodes_from_file(xml_path)

    musical_primitives, banned_ids = remove_staff_lines(musical_primitives)

    barlines = []
    for c in musical_primitives:
        if c.class_name == 'barline' or c.class_name == 'barlineHeavy':
            barlines.append(
                [str(c.id), c.top, c.bottom, c.left, c.right])
    thinbar_df = pd.DataFrame(barlines, columns=['id', 'top', 'bottom', 'left', 'right'])

    # 2- find the consecutive barlines and define each measure
    measures = []
    count_measure = 0
    for i in range(len(thinbar_df)):
        dist2rightbar = -np.inf
        best_candidate = -1
        for j in range(len(thinbar_df)):
            if i != j:
                if abs(thinbar_df.loc[i, 'top'] - thinbar_df.loc[j, 'top']) < 25 and 0 > thinbar_df.loc[i, 'left'] - \
                        thinbar_df.loc[j, 'left'] > dist2rightbar:
                    # If they are on the same line  and close one another
                    dist2rightbar = thinbar_df.loc[i, 'left'] - thinbar_df.loc[j, 'left']
                    best_candidate = j
        if best_candidate != -1:
            thinbar1 = thinbar_df.loc[i]
            thinbar2 = thinbar_df.loc[best_candidate]
            measures.append(
                ['measure_' + str(count_measure), thinbar1.top, thinbar2.bottom, thinbar1.left, thinbar2.right,
                 0, 0])
            count_measure += 1

    measures_df = pd.DataFrame(measures, columns=['id', 'top', 'bottom', 'left', 'right', 'y', 'x'])

    # 2.1 - visualize the measures on the image
    visualize_score_with_obj(measures_df, img_score_path, output_path + "measures/" + score_name + ".png", dpi)

    # 3- associate the object to the measures base on their position and keep track of those linked that might be
    # outside of the measure
    objects_in_measure = [[] for i in range(len(measures))]  # contains for each measure the list of node
    objects_ids_in_measure = [[] for i in range(len(measures))]  # contains for each measure the list of node id
    object_ids_linked_to_measure = [[] for i in range(
        len(measures))]  # contains for each measure the list of node id of object linked to obj inthe measure

    for c in musical_primitives:
        in_n_measure = 0
        for i in range(len(measures)):
            # statement: object should be in the measure height * (1.25) and in the width
            if DATASET == 'muscima-pp':
                height_error = (measures_df.loc[i, 'bottom'] - measures_df.loc[i, 'top']) * .25
            elif DATASET == 'doremi':
                height_error = (measures_df.loc[i, 'bottom'] - measures_df.loc[i, 'top']) * 10
            in_measure = measures_df.loc[i, 'bottom'] + height_error >= c.middle[0] >= measures_df.loc[
                i, 'top'] - height_error and measures_df.loc[i, 'left'] <= c.middle[1] <= measures_df.loc[
                             i, 'right']

            if in_measure:
                in_n_measure += 1
                objects_in_measure[i].append(c)
                objects_ids_in_measure[i].append(str(c.id))
                object_ids_linked_to_measure[i].append(str(c.id))
                for linked_obj in c.outlinks:
                    if linked_obj not in banned_ids:
                        object_ids_linked_to_measure[i].append(str(linked_obj))
                for linked_obj in c.inlinks:
                    if linked_obj not in banned_ids:
                        object_ids_linked_to_measure[i].append(str(linked_obj))
        if in_n_measure < 1:
            # print("Object not in any measure", c.id)
            pass
    # 4- cut the image to the measure taking in account the objects linked to each measure
    # 4.1 find the measure limit
    measures_image_limit = []
    for objects in objects_in_measure:
        if len(objects) > 0:
            min_x = min(c.left for c in objects)
            max_x = max(c.right for c in objects)
            min_y = min(c.top for c in objects)
            max_y = max(c.bottom for c in objects)
        else:
            min_x, max_x, min_y, max_y = 100000, 0, 100000, 0

        measures_image_limit.append([min_x, max_x, min_y, max_y])

    # 4.2 cut image at the limit of the measures and save it
    for img in measures_image_limit:
        if DATASET == 'muscima-pp':
            save_path = f'{output_path}images/fulls/{score_name}_measure_{measures_image_limit.index(img)}.png'
            image = cv2.imread(img_score_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.bitwise_not(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif DATASET == 'doremi':
            save_path = f'{output_path}Images/{score_name}_measure_{measures_image_limit.index(img)}.png'
            image = cv2.imread(img_score_path, cv2.IMREAD_COLOR)

        image = image[img[2]:img[3], img[0]:img[1]]
        w = abs(img[0] - img[1])
        h = abs(img[2] - img[3])

        if (w, h) == (100000, 100000):
            # print("No object in the measure")
            continue

        fig = plt.figure(frameon=False)
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        fig.savefig(save_path, dpi=1)
        plt.close(fig)

    # 5 - generate the XML
    for i in range(len(objects_in_measure)):
        if DATASET == 'muscima-pp':
            output_file_path = f'{output_path}data/annotations/{score_name}_measure_{i}.xml'
        elif DATASET == 'doremi':
            output_file_path = f'{output_path}Parsed_by_measure_omr_xml/{score_name}_measure_{i}.xml'
        filter_xml(xml_path, output_file_path,
                   object_ids_linked_to_measure[i],
                   objects_ids_in_measure[i], measures_image_limit[i],
                   banned_ids)

    # 6 - For a nice visualisation
    df_visu_all_obj = []
    for i in range(len(objects_in_measure)):
        for c in objects_in_measure[i]:
            df_visu_all_obj.append(
                {'id': str(c.id), 'top': c.top, 'bottom': c.bottom, 'left': c.left, 'right': c.right,
                 'y': (c.top + c.bottom) / 2, 'x': (c.left + c.right) / 2, 'measure': i})

    df_visu_all_obj = pd.DataFrame(df_visu_all_obj)
    # Generate colorful full image with all obj
    visualize_score_with_obj(df_visu_all_obj, img_score_path, output_path + 'full_obj/' + score_name + '.png', dpi, measure_obj=len(objects_in_measure))


# Input directory paths
DATASET = 'muscima-pp' # 'doremi' or 'muscima-pp'

if DATASET == 'muscima-pp':
    xml_dir = r'./data/muscima-pp/v2.1/data/annotations/'
    img_dir = r'./data/muscima-pp/v2.1/images/fulls/'
    output_path='./data/muscima-pp/measure_cut/'
elif DATASET == 'doremi':
    xml_dir = r'./data/DoReMi_v1/Parsed_by_page_omr_xml/'
    img_dir = r'./data/DoReMi_v1/Images/'
    output_path = './data/DoReMi_v1/measure_cut/'
else:
    raise ValueError("DATASET must be 'doremi' or 'muscima-pp'")

# Get lists of full paths
list_score_xml_path = [os.path.join(xml_dir, filename) for filename in os.listdir(xml_dir) if filename.endswith('.xml')]
list_score_img_path = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if filename.endswith('.png')]


count_measure = 0
print('Total score to cut: ', len(list_score_xml_path))

if DATASET == 'muscima-pp':
    pairs_xml_img = zip(list_score_xml_path, list_score_img_path)
elif DATASET == 'doremi':
    pairs_xml_img = []
    for xml in list_score_xml_path.copy():
        part1 = re.search(r'_(.*?)-layout', xml.split('/')[-1]).group(1)
        part1 = part1.replace('Parsed_', '')
        page_number = re.search(r'Page_(\d+)', xml.split('/')[-1]).group(1)
        formatted_page_number = f"{int(page_number):03}"
        name_img = f"{part1}-{formatted_page_number}"
        img_path = os.path.join(img_dir, f"{name_img}.png")
        assert img_path in list_score_img_path, f"Image {img_path} not found"
        pairs_xml_img.append((xml, img_path))

for xml_path, image_path in (pbar := tqdm(pairs_xml_img)):
    pbar.set_description(f"Cutting measure ")
    cut_measures(img_score_path=image_path, xml_path=xml_path, output_path=output_path)
    count_measure += 1