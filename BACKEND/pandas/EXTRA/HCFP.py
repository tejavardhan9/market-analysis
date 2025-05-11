import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

# Load dataset
df = pd.read_csv("C:/Users/tejav/OneDrive/Documents/GitHub/market-analysis/BACKEND/pandas/DATASET/Groceries_dataset.csv")
transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()

# Step 1: Compress transactions based on item frequency
def compress_transactions(transactions: List[List[str]], min_item_frequency: int = 2) -> Tuple[List[List[str]], Dict[str, int]]:
    frequency = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            frequency[item] += 1

    def sort_items(t):
        return sorted([item for item in t if frequency[item] >= min_item_frequency], key=lambda x: (-frequency[x], x))

    compressed = [sort_items(t) for t in transactions if sort_items(t)]
    return compressed, frequency

# Step 2: FP-Tree Node Definition
class TreeNode:
    def __init__(self, item: Optional[str], parent: Optional['TreeNode']):
        self.item = item
        self.count = 1
        self.parent = parent
        self.children: Dict[str, 'TreeNode'] = {}
        self.link = None

    def increment(self, count=1):
        self.count += count

# Step 3: Build FP-Tree from compressed transactions
def build_fp_tree(transactions: List[List[str]], min_support: int) -> Tuple[TreeNode, Dict[str, List[TreeNode]]]:
    header_table: Dict[str, List[TreeNode]] = defaultdict(list)
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
        if sum(n.count for n in nodes) >= min_support
    }

    return root, header_table

# Step 4: Mine Frequent Patterns from the FP-tree
def ascend_fp_tree(node: TreeNode) -> List[str]:
    path = []
    while node.parent and node.parent.item is not None:
        node = node.parent
        path.append(node.item)
    return path[::-1]

def mine_patterns(header_table: Dict[str, List[TreeNode]], min_support: int) -> Dict[Tuple[str, ...], int]:
    patterns = {}
    for item, nodes in header_table.items():
        for node in nodes:
            path = ascend_fp_tree(node)
            if path:
                pattern = tuple(sorted(path + [item]))
                patterns[pattern] = patterns.get(pattern, 0) + node.count
    return {pattern: count for pattern, count in patterns.items() if count >= min_support}

# Step 5: Generate Recommendations from Patterns
def generate_recommendations(patterns: Dict[Tuple[str, ...], int]) -> List[Tuple[Tuple[str, ...], int]]:
    return sorted(patterns.items(), key=lambda x: -x[1])

# Step 6: Purchase Trend Analysis (if Date column exists)
def analyze_purchase_trends(df: pd.DataFrame):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['Hour'] = df['Date'].dt.hour

        # Most active shopping day
        active_day = df['DayOfWeek'].mode()[0]

        # Peak shopping hours
        peak_hours = df.groupby('Hour').size().idxmax()

        # Average basket size
        basket_size = df.groupby('Member_number')['itemDescription'].apply(len).mean()

        return active_day, peak_hours, basket_size
    else:
        return None, None, None

# Step 7: Run the Algorithm
# Compress Transactions
compressed_tx, item_freq = compress_transactions(transactions, min_item_frequency=5)

# Build FP-Tree
fp_root, header_table = build_fp_tree(compressed_tx, min_support=5)

# Mine Frequent Patterns
frequent_patterns = mine_patterns(header_table, min_support=5)

# Generate Recommendations
recommendations = generate_recommendations(frequent_patterns)

# Purchase Trend Analysis (assuming the 'Date' column exists)
active_day, peak_hours, basket_size = analyze_purchase_trends(df)

# Display Results
print("Frequently Bought Together:")
for items, count in recommendations[:10]:
    items_str = " + ".join(items)
    confidence = (count / len(transactions)) * 100
    print(f"Customers who bought {items[0]} also bought {items[1]} ({confidence:.2f}%).")

print("\nTop Recommendations:")
print("Based on your history, we recommend:\n")
for items, count in recommendations[:10]:
    items_str = " + ".join(items)
    print(f"{items_str} combo")

print("\nPurchase Trends:")
if active_day:
    print(f"Most active shopping day: {active_day}")
    print(f"Peak shopping hours: {peak_hours}:00")
    print(f"Average basket size: {basket_size:.2f} items")
else:
    print("Purchase trend analysis is unavailable (missing 'Date' column).")
