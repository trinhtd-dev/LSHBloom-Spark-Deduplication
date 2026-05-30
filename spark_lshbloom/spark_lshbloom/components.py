from __future__ import annotations


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1

    def groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for value in self.parent:
            groups.setdefault(self.find(value), []).append(value)
        return groups


def duplicate_pairs_to_groups_local(
    pairs_df,
    query_col: str = "query_id",
    candidate_col: str = "candidate_doc_id",
    max_edges_to_collect: int = 2_000_000,
):
    """Build duplicate groups from candidate pairs with local UnionFind.

    This is intended for experiments and medium-scale validation. For very
    large candidate graphs, use a distributed connected-components library such
    as GraphFrames.
    """

    n_edges = pairs_df.count()
    if n_edges > max_edges_to_collect:
        raise RuntimeError(
            f"Too many edges to collect locally: {n_edges}. "
            "Use a distributed connected-components implementation."
        )

    uf = UnionFind()
    for row in pairs_df.select(query_col, candidate_col).toLocalIterator():
        uf.union(str(row[query_col]), str(row[candidate_col]))

    records = []
    for root, doc_ids in uf.groups().items():
        if len(doc_ids) <= 1:
            continue
        canonical_id = min(doc_ids)
        for doc_id in doc_ids:
            records.append((doc_id, canonical_id, root, len(doc_ids)))

    return pairs_df.sparkSession.createDataFrame(
        records,
        schema="doc_id string, canonical_doc_id string, duplicate_group_id string, group_size int",
    )
