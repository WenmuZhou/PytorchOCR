import copy
import numpy as np
import string
from shapely.geometry import LineString, Point, Polygon
import json
import copy
from random import sample

from torchocr.utils.logging import get_logger


class ClsLabelEncode(object):
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, data):
        label = data['label']
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data['label'] = label
        return data


class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False):

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        if character_dict_path is None:
            logger = get_logger()
            logger.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class E2ELabelEncodeTest(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(E2ELabelEncodeTest, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        import json
        padnum = len(self.dict)
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)
        data['polys'] = boxes
        data['ignore_tags'] = txt_tags
        temp_texts = []
        for text in txts:
            text = text.lower()
            text = self.encode(text)
            if text is None:
                return None
            text = text + [padnum] * (self.max_text_len - len(text)
                                      )  # use 36 to pad
            temp_texts.append(text)
        data['texts'] = np.array(temp_texts)
        return data


class E2ELabelEncodeTrain(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        import json
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data


class KieLabelEncode(object):
    def __init__(self,
                 character_dict_path,
                 class_path,
                 norm=10,
                 directed=False,
                 **kwargs):
        super(KieLabelEncode, self).__init__()
        self.dict = dict({'': 0})
        self.label2classid_map = dict()
        with open(character_dict_path, 'r', encoding='utf-8') as fr:
            idx = 1
            for line in fr:
                char = line.strip()
                self.dict[char] = idx
                idx += 1
        with open(class_path, "r") as fin:
            lines = fin.readlines()
            for idx, line in enumerate(lines):
                line = line.strip("\n")
                self.label2classid_map[line] = idx
        self.norm = norm
        self.directed = directed

    def compute_relation(self, boxes):
        """Compute relation between every two boxes."""
        x1s, y1s = boxes[:, 0:1], boxes[:, 1:2]
        x2s, y2s = boxes[:, 4:5], boxes[:, 5:6]
        ws, hs = x2s - x1s + 1, np.maximum(y2s - y1s + 1, 1)
        dxs = (x1s[:, 0][None] - x1s) / self.norm
        dys = (y1s[:, 0][None] - y1s) / self.norm
        xhhs, xwhs = hs[:, 0][None] / hs, ws[:, 0][None] / hs
        whs = ws / hs + np.zeros_like(xhhs)
        relations = np.stack([dxs, dys, whs, xhhs, xwhs], -1)
        bboxes = np.concatenate([x1s, y1s, x2s, y2s], -1).astype(np.float32)
        return relations, bboxes

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        max_len = 300
        recoder_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = -np.ones((len(text_inds), max_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds, recoder_len

    def list_to_numpy(self, ann_infos):
        """Convert bboxes, relations, texts and labels to ndarray."""
        boxes, text_inds = ann_infos['points'], ann_infos['text_inds']
        boxes = np.array(boxes, np.int32)
        relations, bboxes = self.compute_relation(boxes)

        labels = ann_infos.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = ann_infos.get('edges', None)
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)
        padded_text_inds, recoder_len = self.pad_text_indices(text_inds)
        max_num = 300
        temp_bboxes = np.zeros([max_num, 4])
        h, _ = bboxes.shape
        temp_bboxes[:h, :] = bboxes

        temp_relations = np.zeros([max_num, max_num, 5])
        temp_relations[:h, :h, :] = relations

        temp_padded_text_inds = np.zeros([max_num, max_num])
        temp_padded_text_inds[:h, :] = padded_text_inds

        temp_labels = np.zeros([max_num, max_num])
        temp_labels[:h, :h + 1] = labels

        tag = np.array([h, recoder_len])
        return dict(
            image=ann_infos['image'],
            points=temp_bboxes,
            relations=temp_relations,
            texts=temp_padded_text_inds,
            labels=temp_labels,
            tag=tag)

    def convert_canonical(self, points_x, points_y):

        assert len(points_x) == 4
        assert len(points_y) == 4

        points = [Point(points_x[i], points_y[i]) for i in range(4)]

        polygon = Polygon([(p.x, p.y) for p in points])
        min_x, min_y, _, _ = polygon.bounds
        points_to_lefttop = [
            LineString([points[i], Point(min_x, min_y)]) for i in range(4)
        ]
        distances = np.array([line.length for line in points_to_lefttop])
        sort_dist_idx = np.argsort(distances)
        lefttop_idx = sort_dist_idx[0]

        if lefttop_idx == 0:
            point_orders = [0, 1, 2, 3]
        elif lefttop_idx == 1:
            point_orders = [1, 2, 3, 0]
        elif lefttop_idx == 2:
            point_orders = [2, 3, 0, 1]
        else:
            point_orders = [3, 0, 1, 2]

        sorted_points_x = [points_x[i] for i in point_orders]
        sorted_points_y = [points_y[j] for j in point_orders]

        return sorted_points_x, sorted_points_y

    def sort_vertex(self, points_x, points_y):

        assert len(points_x) == 4
        assert len(points_y) == 4

        x = np.array(points_x)
        y = np.array(points_y)
        center_x = np.sum(x) * 0.25
        center_y = np.sum(y) * 0.25

        x_arr = np.array(x - center_x)
        y_arr = np.array(y - center_y)

        angle = np.arctan2(y_arr, x_arr) * 180.0 / np.pi
        sort_idx = np.argsort(angle)

        sorted_points_x, sorted_points_y = [], []
        for i in range(4):
            sorted_points_x.append(points_x[sort_idx[i]])
            sorted_points_y.append(points_y[sort_idx[i]])

        return self.convert_canonical(sorted_points_x, sorted_points_y)

    def __call__(self, data):
        import json
        label = data['label']
        annotations = json.loads(label)
        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        for ann in annotations:
            box = ann['points']
            x_list = [box[i][0] for i in range(4)]
            y_list = [box[i][1] for i in range(4)]
            sorted_x_list, sorted_y_list = self.sort_vertex(x_list, y_list)
            sorted_box = []
            for x, y in zip(sorted_x_list, sorted_y_list):
                sorted_box.append(x)
                sorted_box.append(y)
            boxes.append(sorted_box)
            text = ann['transcription']
            texts.append(ann['transcription'])
            text_ind = [self.dict[c] for c in text if c in self.dict]
            text_inds.append(text_ind)
            if 'label' in ann.keys():
                labels.append(self.label2classid_map[ann['label']])
            elif 'key_cls' in ann.keys():
                labels.append(ann['key_cls'])
            else:
                raise ValueError(
                    "Cannot found 'key_cls' in ann.keys(), please check your training annotation."
                )
            edges.append(ann.get('edge', 0))
        ann_infos = dict(
            image=data['image'],
            points=boxes,
            texts=texts,
            text_inds=text_inds,
            edges=edges,
            labels=labels)

        return self.list_to_numpy(ann_infos)


class AttnLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(AttnLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text = [0] + text + [len(self.character) - 1] + [0] * (self.max_text_len
                                                               - len(text) - 2)
        data['label'] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class RFLLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(RFLLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def encode_cnt(self, text):
        cnt_label = [0.0] * len(self.character)
        for char_ in text:
            cnt_label[char_] += 1
        return np.array(cnt_label)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        cnt_label = self.encode_cnt(text)
        data['length'] = np.array(len(text))
        text = [0] + text + [len(self.character) - 1] + [0] * (self.max_text_len
                                                               - len(text) - 2)
        if len(text) != self.max_text_len:
            return None
        data['label'] = np.array(text)
        data['cnt_label'] = cnt_label
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SEEDLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(SEEDLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.padding = "padding"
        self.end_str = "eos"
        self.unknown = "unknown"
        dict_character = dict_character + [
            self.end_str, self.padding, self.unknown
        ]
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text)) + 1  # conclude eos
        text = text + [len(self.character) - 3] + [len(self.character) - 2] * (
            self.max_text_len - len(text) - 1)
        data['label'] = np.array(text)
        return data


class SRNLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length=25,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(SRNLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        char_num = len(self.character)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text = text + [char_num - 1] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class TableLabelEncode(AttnLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path,
                 replace_empty_cell_token=False,
                 merge_no_span_structure=False,
                 learn_empty_box=False,
                 loc_reg_num=4,
                 **kwargs):
        self.max_text_len = max_text_length
        self.lower = False
        self.learn_empty_box = learn_empty_box
        self.merge_no_span_structure = merge_no_span_structure
        self.replace_empty_cell_token = replace_empty_cell_token

        dict_character = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                dict_character.append(line)

        if self.merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.idx2char = {v: k for k, v in self.dict.items()}

        self.character = dict_character
        self.loc_reg_num = loc_reg_num
        self.pad_idx = self.dict[self.beg_str]
        self.start_idx = self.dict[self.beg_str]
        self.end_idx = self.dict[self.end_str]

        self.td_token = ['<td>', '<td', '<eb></eb>', '<td></td>']
        self.empty_bbox_token_dict = {
            "[]": '<eb></eb>',
            "[' ']": '<eb1></eb1>',
            "['<b>', ' ', '</b>']": '<eb2></eb2>',
            "['\\u2028', '\\u2028']": '<eb3></eb3>',
            "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
            "['<b>', '</b>']": '<eb5></eb5>',
            "['<i>', ' ', '</i>']": '<eb6></eb6>',
            "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
            "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
            "['<i>', '</i>']": '<eb9></eb9>',
            "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']":
            '<eb10></eb10>',
        }

    @property
    def _max_text_len(self):
        return self.max_text_len + 2

    def __call__(self, data):
        cells = data['cells']
        structure = data['structure']
        if self.merge_no_span_structure:
            structure = self._merge_no_span_structure(structure)
        if self.replace_empty_cell_token:
            structure = self._replace_empty_cell_token(structure, cells)
        # remove empty token and add " " to span token
        new_structure = []
        for token in structure:
            if token != '':
                if 'span' in token and token[0] != ' ':
                    token = ' ' + token
                new_structure.append(token)
        # encode structure
        structure = self.encode(new_structure)
        if structure is None:
            return None

        structure = [self.start_idx] + structure + [self.end_idx
                                                    ]  # add sos abd eos
        structure = structure + [self.pad_idx] * (self._max_text_len -
                                                  len(structure))  # pad
        structure = np.array(structure)
        data['structure'] = structure

        if len(structure) > self._max_text_len:
            return None

        # encode box
        bboxes = np.zeros(
            (self._max_text_len, self.loc_reg_num), dtype=np.float32)
        bbox_masks = np.zeros((self._max_text_len, 1), dtype=np.float32)

        bbox_idx = 0

        for i, token in enumerate(structure):
            if self.idx2char[token] in self.td_token:
                if 'bbox' in cells[bbox_idx] and len(cells[bbox_idx][
                        'tokens']) > 0:
                    bbox = cells[bbox_idx]['bbox'].copy()
                    bbox = np.array(bbox, dtype=np.float32).reshape(-1)
                    bboxes[i] = bbox
                    bbox_masks[i] = 1.0
                if self.learn_empty_box:
                    bbox_masks[i] = 1.0
                bbox_idx += 1
        data['bboxes'] = bboxes
        data['bbox_masks'] = bbox_masks
        return data

    def _merge_no_span_structure(self, structure):
        """
        This code is refer from:
        https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
        """
        new_structure = []
        i = 0
        while i < len(structure):
            token = structure[i]
            if token == '<td>':
                token = '<td></td>'
                i += 1
            new_structure.append(token)
            i += 1
        return new_structure

    def _replace_empty_cell_token(self, token_list, cells):
        """
        This fun code is refer from:
        https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
        """

        bbox_idx = 0
        add_empty_bbox_token_list = []
        for token in token_list:
            if token in ['<td></td>', '<td', '<td>']:
                if 'bbox' not in cells[bbox_idx].keys():
                    content = str(cells[bbox_idx]['tokens'])
                    token = self.empty_bbox_token_dict[content]
                add_empty_bbox_token_list.append(token)
                bbox_idx += 1
            else:
                add_empty_bbox_token_list.append(token)
        return add_empty_bbox_token_list


class TableMasterLabelEncode(TableLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path,
                 replace_empty_cell_token=False,
                 merge_no_span_structure=False,
                 learn_empty_box=False,
                 loc_reg_num=4,
                 **kwargs):
        super(TableMasterLabelEncode, self).__init__(
            max_text_length, character_dict_path, replace_empty_cell_token,
            merge_no_span_structure, learn_empty_box, loc_reg_num, **kwargs)
        self.pad_idx = self.dict[self.pad_str]
        self.unknown_idx = self.dict[self.unknown_str]

    @property
    def _max_text_len(self):
        return self.max_text_len

    def add_special_char(self, dict_character):
        self.beg_str = '<SOS>'
        self.end_str = '<EOS>'
        self.unknown_str = '<UKN>'
        self.pad_str = '<PAD>'
        dict_character = dict_character
        dict_character = dict_character + [
            self.unknown_str, self.beg_str, self.end_str, self.pad_str
        ]
        return dict_character


class TableBoxEncode(object):
    def __init__(self, in_box_format='xyxy', out_box_format='xyxy', **kwargs):
        assert out_box_format in ['xywh', 'xyxy', 'xyxyxyxy']
        self.in_box_format = in_box_format
        self.out_box_format = out_box_format

    def __call__(self, data):
        img_height, img_width = data['image'].shape[:2]
        bboxes = data['bboxes']
        if self.in_box_format != self.out_box_format:
            if self.out_box_format == 'xywh':
                if self.in_box_format == 'xyxyxyxy':
                    bboxes = self.xyxyxyxy2xywh(bboxes)
                elif self.in_box_format == 'xyxy':
                    bboxes = self.xyxy2xywh(bboxes)

        bboxes[:, 0::2] /= img_width
        bboxes[:, 1::2] /= img_height
        data['bboxes'] = bboxes
        return data

    def xyxyxyxy2xywh(self, bboxes):
        new_bboxes = np.zeros([len(bboxes), 4])
        new_bboxes[:, 0] = bboxes[:, 0::2].min()  # x1
        new_bboxes[:, 1] = bboxes[:, 1::2].min()  # y1
        new_bboxes[:, 2] = bboxes[:, 0::2].max() - new_bboxes[:, 0]  # w
        new_bboxes[:, 3] = bboxes[:, 1::2].max() - new_bboxes[:, 1]  # h
        return new_bboxes

    def xyxy2xywh(self, bboxes):
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x center
        new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y center
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
        return new_bboxes


class SARLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(SARLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data['length'] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[:len(target)] = target
        data['label'] = np.array(padded_text)
        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]


class SATRNLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False,
                 **kwargs):
        super(SATRNLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        self.lower = lower

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    def encode(self, text):
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            text_list.append(self.dict.get(char, self.unknown_idx))
        if len(text_list) == 0:
            return None
        return text_list

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]
        if len(target) > self.max_text_len:
            padded_text = target[:self.max_text_len]
        else:
            padded_text[:len(target)] = target
        data['label'] = np.array(padded_text)
        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]


class PRENLabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path,
                 use_space_char=False,
                 **kwargs):
        super(PRENLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        padding_str = '<PAD>'  # 0 
        end_str = '<EOS>'  # 1
        unknown_str = '<UNK>'  # 2

        dict_character = [padding_str, end_str, unknown_str] + dict_character
        self.padding_idx = 0
        self.end_idx = 1
        self.unknown_idx = 2

        return dict_character

    def encode(self, text):
        if len(text) == 0 or len(text) >= self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                text_list.append(self.unknown_idx)
            else:
                text_list.append(self.dict[char])
        text_list.append(self.end_idx)
        if len(text_list) < self.max_text_len:
            text_list += [self.padding_idx] * (
                self.max_text_len - len(text_list))
        return text_list

    def __call__(self, data):
        text = data['label']
        encoded_text = self.encode(text)
        if encoded_text is None:
            return None
        data['label'] = np.array(encoded_text)
        return data

class MultiLabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 gtc_encode=None,
                 **kwargs):
        super(MultiLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

        self.ctc_encode = CTCLabelEncode(max_text_length, character_dict_path,
                                         use_space_char, **kwargs)
        self.gtc_encode_type = gtc_encode
        if gtc_encode is None:
            self.gtc_encode = SARLabelEncode(
                max_text_length, character_dict_path, use_space_char, **kwargs)
        else:
            self.gtc_encode = eval(gtc_encode)(
                max_text_length, character_dict_path, use_space_char, **kwargs)

    def __call__(self, data):
        data_ctc = copy.deepcopy(data)
        data_gtc = copy.deepcopy(data)
        data_out = dict()
        data_out['img_path'] = data.get('img_path', None)
        data_out['image'] = data['image']
        ctc = self.ctc_encode.__call__(data_ctc)
        gtc = self.gtc_encode.__call__(data_gtc)
        if ctc is None or gtc is None:
            return None
        data_out['label_ctc'] = ctc['label']
        if self.gtc_encode_type is not None:
            data_out['label_gtc'] = gtc['label']
        else:
            data_out['label_sar'] = gtc['label']
        data_out['length'] = ctc['length']
        return data_out


class NRTRLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):

        super(NRTRLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data['length'] = np.array(len(text))
        text.insert(0, 2)
        text.append(3)
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        return dict_character


class ViTSTRLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 ignore_index=0,
                 **kwargs):

        super(ViTSTRLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        self.ignore_index = ignore_index

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text.insert(0, self.ignore_index)
        text.append(1)
        text = text + [self.ignore_index] * (self.max_text_len + 2 - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['<s>', '</s>'] + dict_character
        return dict_character


class ABINetLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 ignore_index=100,
                 **kwargs):

        super(ABINetLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        self.ignore_index = ignore_index

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        text.append(0)
        text = text + [self.ignore_index] * (self.max_text_len + 1 - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        return dict_character


class SRLabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(SRLabelEncode, self).__init__(max_text_length,
                                            character_dict_path, use_space_char)
        self.dic = {}
        with open(character_dict_path, 'r') as fin:
            for line in fin.readlines():
                line = line.strip()
                character, sequence = line.split()
                self.dic[character] = sequence
        english_stroke_alphabet = '0123456789'
        self.english_stroke_dict = {}
        for index in range(len(english_stroke_alphabet)):
            self.english_stroke_dict[english_stroke_alphabet[index]] = index

    def encode(self, label):
        stroke_sequence = ''
        for character in label:
            if character not in self.dic:
                continue
            else:
                stroke_sequence += self.dic[character]
        stroke_sequence += '0'
        label = stroke_sequence

        length = len(label)

        input_tensor = np.zeros(self.max_text_len).astype("int64")
        for j in range(length - 1):
            input_tensor[j + 1] = self.english_stroke_dict[label[j]]

        return length, input_tensor

    def __call__(self, data):
        text = data['label']
        length, input_tensor = self.encode(text)

        data["length"] = length
        data["input_tensor"] = input_tensor
        if text is None:
            return None
        return data


class SPINLabelEncode(AttnLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=True,
                 **kwargs):
        super(SPINLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)
        self.lower = lower

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + [self.end_str] + dict_character
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None
        data['length'] = np.array(len(text))
        target = [0] + text + [1]
        padded_text = [0 for _ in range(self.max_text_len + 2)]

        padded_text[:len(target)] = target
        data['label'] = np.array(padded_text)
        return data


class VLLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(VLLabelEncode, self).__init__(max_text_length,
                                            character_dict_path, use_space_char)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def __call__(self, data):
        text = data['label']  # original string
        # generate occluded text
        len_str = len(text)
        if len_str <= 0:
            return None
        change_num = 1
        order = list(range(len_str))
        change_id = sample(order, change_num)[0]
        label_sub = text[change_id]
        if change_id == (len_str - 1):
            label_res = text[:change_id]
        elif change_id == 0:
            label_res = text[1:]
        else:
            label_res = text[:change_id] + text[change_id + 1:]

        data['label_res'] = label_res  # remaining string
        data['label_sub'] = label_sub  # occluded character
        data['label_id'] = change_id  # character index
        # encode label
        text = self.encode(text)
        if text is None:
            return None
        text = [i + 1 for i in text]
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        label_res = self.encode(label_res)
        label_sub = self.encode(label_sub)
        if label_res is None:
            label_res = []
        else:
            label_res = [i + 1 for i in label_res]
        if label_sub is None:
            label_sub = []
        else:
            label_sub = [i + 1 for i in label_sub]
        data['length_res'] = np.array(len(label_res))
        data['length_sub'] = np.array(len(label_sub))
        label_res = label_res + [0] * (self.max_text_len - len(label_res))
        label_sub = label_sub + [0] * (self.max_text_len - len(label_sub))
        data['label_res'] = np.array(label_res)
        data['label_sub'] = np.array(label_sub)
        return data


class CTLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = data['label']

        label = json.loads(label)
        nBox = len(label)
        boxes, txts = [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            box = np.array(box)

            boxes.append(box)
            txt = label[bno]['transcription']
            txts.append(txt)

        if len(boxes) == 0:
            return None

        data['polys'] = boxes
        data['texts'] = txts
        return data


class CANLabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 character_dict_path,
                 max_text_length=100,
                 use_space_char=False,
                 lower=True,
                 **kwargs):
        super(CANLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char, lower)

    def encode(self, text_seq):
        text_seq_encoded = []
        for text in text_seq:
            if text not in self.character:
                continue
            text_seq_encoded.append(self.dict.get(text))
        if len(text_seq_encoded) == 0:
            return None
        return text_seq_encoded

    def __call__(self, data):
        label = data['label']
        if isinstance(label, str):
            label = label.strip().split()
        label.append(self.end_str)
        data['label'] = self.encode(label)
        return data
