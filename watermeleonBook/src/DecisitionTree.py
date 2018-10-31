from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import math
import copy
from abc import abstractmethod


class SplitStandard(object):
    def __init__(self):
        pass

    @abstractmethod
    def calcu_value(self, y):
        pass


class EntropySplitStandard(SplitStandard):

    def __init__(self):
        super().__init__()

    def calcu_value(self, y):
        values, counts = np.unique(y, return_counts=True)
        cur_entropy = stats.entropy(counts/sum(counts), base=2)
        return cur_entropy


class GiniSplitStandard(SplitStandard):
    def __init__(self):
        super().__init__()

    def calcu_value(self, y):
        pass

class TreeNode(object):

    def __init__(self, set_x, set_y, label, delta_entropy, entropy):
        self.children_nodes = []
        self.data_set = set_x
        self.result_set = set_y
        self.label = label
        self.is_leaf = False
        self.delta_entropy = delta_entropy
        self.entropy = DecisitionTree.calcu_entropy(set_y)

    def create_child_node(self, node):
        self.children_nodes.append(node)

    def set_leaf(self, is_leaf):
        self.is_leaf = is_leaf

    def set_label(self, label):
        self.label = label

    def set_best_column(self, best_column):
        self.best_column = best_column

class DecisitionTree(object):

    def __init__(self):
        self.split_method = EntropySplitStandard()

    def data_similar(self, training_set_x):
        for column_name in training_set_x.columns:
            if(len(np.unique(training_set_x.loc[:, column_name])) == 1):
                continue
            else:
                return False

        return True

    def get_best_num_split_point(self, cand_split_points, vs, df_y):
        min_entropy = math.pow(2, 63) - 1
        best_point = cand_split_points.iloc[0]
        for point in cand_split_points:
            cur_entropy = (len(vs[vs >= point]) / len(vs)) * self.split_method.calcu_value(df_y[vs >= point]) + (len(vs[vs < point]) / len(vs)) * self.split_method.calcu_value(df_y[vs < point])
            if min_entropy > cur_entropy:
                min_entropy, best_point = cur_entropy, point

        return best_point, min_entropy

    def get_best_attr(self, training_set_x, training_set_y, columnList):
        cur_entropy = self.split_method.calcu_value(training_set_y)
        delta_entropy_map = defaultdict()
        for column_name in columnList:
            column_type = type(training_set_x.loc[:, column_name].iloc[0])
            if column_type == str:
                entropy_sum = 0
                row_types = np.unique(training_set_x[column_name])
                for row_type in row_types:
                    percent = len(training_set_y[training_set_x[column_name] == row_type]) / len(training_set_y)
                    entropy_sum += percent * self.split_method.calcu_value(training_set_y[training_set_x[column_name] == row_type])
                delta_entropy_map[column_name] = [cur_entropy - entropy_sum, column_type, None]
            else:
                sorted_training_set_x = training_set_x.sort_values(by=column_name, ascending=True)
                vs = sorted_training_set_x[column_name]
                candidates_split_points = np.add(vs[:-1], vs[1:])/2

                best_num_split_point, entropy = self.get_best_num_split_point(candidates_split_points, vs, training_set_y)
                delta_entropy_map[column_name] = [cur_entropy - entropy, column_type, best_num_split_point]


        max_entropy = -1 * math.pow(2, 63) - 1
        max_key = None
        max_type = None
        best_point = None
        for key, value in delta_entropy_map.items():
            if value[0] > max_entropy:
                max_entropy = value[0]
                max_key = key
                max_type = value[1]
                best_point = value[2]

        return max_key, max_type, best_point, max_entropy

    def generateTree(self, cur_node, training_set_x, training_set_y, column_list):
        new_copy_list = copy.copy(column_list)
        if(len(training_set_y) != 0 and len(training_set_y.unique()) == 1):
            cur_node.set_label(training_set_y.iloc[0])
            cur_node.set_leaf(True)
            return
        if(len(column_list) == 0 or self.data_similar(training_set_x)):
            cur_node.set_label(stats.mode(training_set_y).mode[0])
            cur_node.set_leaf(True)
            return

        best_column, type, split_point, delta_entropy = self.get_best_attr(training_set_x,
                                       training_set_y, new_copy_list)
        cur_node.set_best_column(best_column)
        if type == str:
            row_values = np.unique(training_set_x[best_column])
            for row_value in row_values:
                indexes = training_set_x[best_column] == row_value
                child_node = TreeNode(training_set_x[indexes], training_set_y[indexes],
                                      None, delta_entropy, self.split_method.calcu_value(training_set_y[indexes]))
                cur_node.create_child_node(child_node)
                if len(indexes) == 0:
                    child_node.set_leaf(True)
                    child_node.set_label(stats.mode(training_set_y).mode[0])
                    return
                else:
                    new_copy_list.remove(best_column)
                    self.generateTree(child_node, training_set_x[indexes], training_set_y[indexes], new_copy_list)
                    new_copy_list.append(best_column)
            new_copy_list.remove(best_column)

        else:
            indexes_large = training_set_x[best_column] >= split_point
            indexes_small = training_set_x[best_column] < split_point
            index_list = [indexes_small, indexes_large]
            for indexes in index_list:
                child_node = TreeNode(training_set_x[indexes], training_set_y[indexes],
                                      None, delta_entropy, self.split_method.calcu_value(training_set_y[indexes]))
                cur_node.create_child_node(child_node)
                if len(indexes) == 0:
                    child_node.set_leaf(True)
                    child_node.set_label(stats.mode(training_set_y).mode[0])
                    return
                else:
                    self.generateTree(child_node, training_set_x[indexes], training_set_y[indexes], new_copy_list)

    # @staticmethod
    # def calcu_entropy(y):
    #     values, counts = np.unique(y, return_counts=True)
    #
    #     cur_entropy = stats.entropy(counts/sum(counts), base=2)
    #     return cur_entropy

    def train(self, training_set_x = pd.DataFrame(), training_set_y = pd.Series(),
              testing_set_x = pd.DataFrame(), testing_set_y = pd.Series()):

        self.root_node = TreeNode(training_set_x, training_set_y, None, self.split_method.calcu_value(training_set_y)
                                  , self.split_method.calcu_value(training_set_y))
        self.generateTree(self.root_node, training_set_x, training_set_y, pd.Series.tolist(pd.Series(training_set_x.columns)))
        print(self.root_node)



if __name__ == '__main__':
    decisitionTree = DecisitionTree()

    df = pd.read_csv('../watermeleonDataSet/table_4_3.csv')
    decisitionTree.train(training_set_x=df.iloc[:, 1:-1], training_set_y=df.iloc[:, -1])

