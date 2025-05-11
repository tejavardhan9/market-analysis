import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

class TreeNode:
    def __init__(self, item: Optional[str], parent: Optional['TreeNode']):
        self.item = item
        self.count = 1
        self.parent = parent
        self.children: Dict[str, 'TreeNode'] = {}
        self.link = None

    def increment(self, count=1):
        self.count += count

class HCFPGrowthModel:
    def __init__(self, min_item_frequency: int = 5, min_support: int = 5):
        self.min_item_frequency = min_item_frequency
        self.min_support = min_support
        self.recommendations: List[Tuple[Tuple[str, ...], int]] = []
        self.trends = ("Unknown", 0, 0.0)

    def fit(self, df: pd.DataFrame):
        self.df = df
        transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()

        compressed_tx, _ = self._compress_transactions(transactions)
        root, header_table = self._build_fp_tree(compressed_tx)

        patterns = self._mine_patterns(header_table)
        self.recommendations = self._generate_recommendations(patterns)

        self.trends = self._analyze_purchase_trends(df)

    def recommend(self, top_n: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        return self.recommendations[:top_n]

    def get_trends(self) -> Tuple[str, int, float]:
        return self.trends

    def _compress_transactions(self, transactions: List[List[str]]) -> Tuple[List[List[str]], Dict[str, int]]:
        frequency = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                frequency[item] += 1

        def sort_items(t):
            return sorted([item for item in t if frequency[item] >= self.min_item_frequency],
                          key=lambda x: (-frequency[x], x))

        compressed = [sort_items(t) for t in transactions if sort_items(t)]
        return compressed, frequency

    def _build_fp_tree(self, transactions: List[List[str]]) -> Tuple[TreeNode, Dict[str, List[TreeNode]]]:
        header_table = defaultdict(list)
        root = TreeNode(None, None)

        for transaction in transactions:
            current_node = root
            for item in transaction:
                if item in current_node.children:
                    current_node.children[item].increment()
                else:
                    new_node = TreeNode(item, current_node)
                    current_node.children[item] = new_node
                    header_table[item].append(new_node)
                current_node = current_node.children[item]

        header_table = {
            item: nodes for item, nodes in header_table.items()
            if sum(n.count for n in nodes) >= self.min_support
        }

        return root, header_table

    def _ascend_fp_tree(self, node: TreeNode) -> List[str]:
        path = []
        while node.parent and node.parent.item is not None:
            node = node.parent
            path.append(node.item)
        return path[::-1]

    def _mine_patterns(self, header_table: Dict[str, List[TreeNode]]) -> Dict[Tuple[str, ...], int]:
        patterns = {}
        for item, nodes in header_table.items():
            for node in nodes:
                path = self._ascend_fp_tree(node)
                if path:
                    pattern = tuple(sorted(path + [item]))
                    patterns[pattern] = patterns.get(pattern, 0) + node.count
        return {pattern: count for pattern, count in patterns.items() if count >= self.min_support}

    def _generate_recommendations(self, patterns: Dict[Tuple[str, ...], int]) -> List[Tuple[Tuple[str, ...], int]]:
        return sorted(patterns.items(), key=lambda x: -x[1])

    def _analyze_purchase_trends(self, df: pd.DataFrame) -> Tuple[str, int, float]:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['DayOfWeek'] = df['Date'].dt.day_name()
            df['Hour'] = df['Date'].dt.hour

            active_day = df['DayOfWeek'].mode()[0]
            peak_hour = df.groupby('Hour').size().idxmax()
            basket_size = df.groupby('Member_number')['itemDescription'].apply(len).mean()

            return active_day, peak_hour, round(basket_size, 2)
        else:
            return "N/A", 0, 0.0
