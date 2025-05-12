import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

class HCFPGrowthModel:
    def __init__(self, min_item_frequency=5, min_support=5):
        self.min_item_frequency = min_item_frequency
        self.min_support = min_support
        self.frequent_patterns = {}
        self.recommendations = []
        self.item_frequency = {}
        self.active_day = None
        self.peak_hours = None
        self.basket_size = None

    class TreeNode:
        def __init__(self, item: Optional[str], parent: Optional['HCFPGrowthModel.TreeNode']):
            self.item = item
            self.count = 1
            self.parent = parent
            self.children: Dict[str, 'HCFPGrowthModel.TreeNode'] = {}
            self.link = None

        def increment(self, count=1):
            self.count += count

    def _compress_transactions(self, transactions: List[List[str]]) -> List[List[str]]:
        frequency = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                frequency[item] += 1
        self.item_frequency = dict(frequency)

        def sort_items(t):
            return sorted(
                [item for item in t if frequency[item] >= self.min_item_frequency],
                key=lambda x: (-frequency[x], x)
            )

        return [sort_items(t) for t in transactions if sort_items(t)]

    def _build_fp_tree(self, transactions: List[List[str]]) -> Tuple['TreeNode', Dict[str, List['TreeNode']]]:
        header_table: Dict[str, List[HCFPGrowthModel.TreeNode]] = defaultdict(list)
        root = self.TreeNode(None, None)

        for transaction in transactions:
            current_node = root
            for item in transaction:
                if item in current_node.children:
                    current_node.children[item].increment()
                else:
                    new_node = self.TreeNode(item, current_node)
                    current_node.children[item] = new_node
                    header_table[item].append(new_node)
                current_node = current_node.children[item]

        header_table = {
            item: nodes for item, nodes in header_table.items()
            if sum(n.count for n in nodes) >= self.min_support
        }
        return root, header_table

    def _ascend_fp_tree(self, node: 'TreeNode') -> List[str]:
        path = []
        while node.parent and node.parent.item is not None:
            node = node.parent
            path.append(node.item)
        return path[::-1]

    def _mine_patterns(self, header_table: Dict[str, List['TreeNode']]) -> Dict[Tuple[str, ...], int]:
        patterns = {}
        for item, nodes in header_table.items():
            for node in nodes:
                path = self._ascend_fp_tree(node)
                if path:
                    pattern = tuple(sorted(path + [item]))
                    patterns[pattern] = patterns.get(pattern, 0) + node.count
        return {p: c for p, c in patterns.items() if c >= self.min_support}

    def fit(self, df: pd.DataFrame):
        transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()
        compressed = self._compress_transactions(transactions)
        _, header_table = self._build_fp_tree(compressed)
        self.frequent_patterns = self._mine_patterns(header_table)
        self.recommendations = sorted(self.frequent_patterns.items(), key=lambda x: -x[1])
        self._analyze_trends(df)

    def recommend(self, top_n=10) -> List[Tuple[Tuple[str, ...], int]]:
        return self.recommendations[:top_n]

    def _analyze_trends(self, df: pd.DataFrame):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DayOfWeek'] = df['Date'].dt.day_name()
            df['Hour'] = df['Date'].dt.hour

            self.active_day = df['DayOfWeek'].mode()[0]
            self.peak_hours = df.groupby('Hour').size().idxmax()
            self.basket_size = df.groupby('Member_number')['itemDescription'].apply(len).mean()

    def get_trends(self) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        return self.active_day, self.peak_hours, self.basket_size
