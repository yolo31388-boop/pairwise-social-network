"""
游戏内玩家社群与社交网络系统 - 核心模块

功能：
- build_graph(): 构建带关系类型与关系强度的社交网络图
- influence_propagation(): 加权级联影响力传播模型（含距离衰减）
- community_detection(): 标签传播社群发现算法 + 模块度质量评估
- social_evolution(): 社群关系动态演化 + 未来状态预测
- recommendation(): 协同过滤好友/社群推荐 + 新玩家冷启动
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
import math
import time


# 支持的关系类型：好友、师徒、夫妻、敌人
RELATION_TYPES = ("friend", "mentor", "spouse", "enemy")
# 正向关系类型（enemy 为负向关系）
POSITIVE_TYPES = ("friend", "mentor", "spouse")


@dataclass
class Config:
    decay: float = 0.5
    influence_threshold: float = 0.1
    growth_rate: float = 0.1
    decay_rate: float = 0.05


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class SocialNetwork:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.players: Dict[str, Dict] = {}
        # (小id, 大id) -> {关系类型: 强度}
        self.relations: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._graph: Optional[Dict] = None
        self._state = {}
        self._history: List[Dict] = []
        self._time = 0.0

    # ------------------------------------------------------------------
    # 基础数据维护
    # ------------------------------------------------------------------
    def add_player(self, player_id: str, **attrs):
        self.players.setdefault(player_id, {}).update(attrs)
        self._graph = None

    def add_relation(self, a: str, b: str, relation_type: str = "friend",
                     strength: float = 0.5):
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"未知关系类型: {relation_type}")
        if a == b:
            raise ValueError("不能与自己建立关系")
        self.add_player(a)
        self.add_player(b)
        key = (a, b) if a < b else (b, a)
        self.relations.setdefault(key, {})[relation_type] = _clamp(strength)
        self._graph = None

    @staticmethod
    def _edge_weight(rels: Dict[str, float]) -> float:
        """边权重 = 正向关系强度之和 - 敌对关系强度。"""
        weight = sum(rels.get(t, 0.0) for t in POSITIVE_TYPES)
        weight -= rels.get("enemy", 0.0)
        return weight

    # ------------------------------------------------------------------
    # 社交网络构建：关系类型 + 关系强度
    # ------------------------------------------------------------------
    def build_graph(self) -> Dict:
        nodes = [{"id": pid, **attrs} for pid, attrs in self.players.items()]
        edges = []
        adjacency: Dict[str, Dict[str, float]] = {pid: {} for pid in self.players}
        for (a, b), rels in self.relations.items():
            weight = self._edge_weight(rels)
            edges.append({
                "source": a,
                "target": b,
                "types": dict(rels),
                "weight": weight,
            })
            adjacency.setdefault(a, {})[b] = weight
            adjacency.setdefault(b, {})[a] = weight
        self._graph = {"nodes": nodes, "edges": edges, "adjacency": adjacency}
        return self._graph

    def _adjacency(self) -> Dict[str, Dict[str, float]]:
        if self._graph is None:
            self.build_graph()
        return self._graph["adjacency"]

    # ------------------------------------------------------------------
    # 影响力传播：加权级联模型 + 距离衰减
    # ------------------------------------------------------------------
    def influence_propagation(self, topic: str, seeds: List[str],
                              decay: Optional[float] = None,
                              threshold: Optional[float] = None,
                              max_hops: int = 10) -> Dict:
        decay = self.config.get("decay", 0.5) if decay is None else decay
        threshold = (self.config.get("influence_threshold", 0.1)
                     if threshold is None else threshold)
        adjacency = self._adjacency()

        # 加权级联：score(v) = max_path(score(u) * w(u,v) * decay)
        # 用最大堆做类 Dijkstra 的最优路径传播
        scores: Dict[str, float] = {}
        hops: Dict[str, int] = {}
        heap: List[Tuple[float, str, int]] = []
        for seed in seeds:
            if seed in self.players:
                scores[seed] = 1.0
                hops[seed] = 0
                heapq.heappush(heap, (-1.0, seed, 0))

        while heap:
            neg_score, node, hop = heapq.heappop(heap)
            score = -neg_score
            if score < scores.get(node, 0.0) or hop >= max_hops:
                continue
            for neighbor, weight in adjacency.get(node, {}).items():
                if weight <= 0:  # 敌对等负向关系不传播正面影响力
                    continue
                new_score = score * weight * decay
                if new_score > scores.get(neighbor, 0.0):
                    scores[neighbor] = new_score
                    hops[neighbor] = hop + 1
                    heapq.heappush(heap, (-new_score, neighbor, hop + 1))

        influenced = {pid: round(score, 6)
                      for pid, score in scores.items() if score >= threshold}
        return {
            "topic": topic,
            "model": "weighted_cascade",
            "decay": decay,
            "threshold": threshold,
            "seeds": list(seeds),
            "influenced": influenced,
            "hops": {pid: hops[pid] for pid in influenced},
        }

    # ------------------------------------------------------------------
    # 社群发现：加权标签传播算法 + 模块度评估
    # ------------------------------------------------------------------
    def community_detection(self, max_iter: int = 100) -> Dict:
        adjacency = self._adjacency()
        labels = {pid: pid for pid in self.players}

        for _ in range(max_iter):
            changed = False
            for pid in sorted(self.players):
                label_weights: Dict[str, float] = {}
                for neighbor, weight in adjacency.get(pid, {}).items():
                    if weight <= 0:
                        continue
                    label = labels[neighbor]
                    label_weights[label] = label_weights.get(label, 0.0) + weight
                best_label, best_weight = labels[pid], 0.0
                for label in sorted(label_weights):
                    if label_weights[label] > best_weight:
                        best_label, best_weight = label, label_weights[label]
                if best_label != labels[pid]:
                    labels[pid] = best_label
                    changed = True
            if not changed:
                break

        communities: Dict[str, List[str]] = {}
        for pid, label in labels.items():
            communities.setdefault(label, []).append(pid)
        for members in communities.values():
            members.sort()

        return {
            "algorithm": "label_propagation",
            "communities": communities,
            "modularity": self._modularity(adjacency, labels),
        }

    @staticmethod
    def _modularity(adjacency: Dict[str, Dict[str, float]],
                    labels: Dict[str, str]) -> float:
        """加权模块度 Q：衡量社群划分质量，越接近 1 社群结构越明显。"""
        degree = {pid: sum(w for w in nbrs.values() if w > 0)
                  for pid, nbrs in adjacency.items()}
        total = sum(degree.values())
        if total <= 0:
            return 0.0
        q = 0.0
        for i in adjacency:
            for j in adjacency:
                if labels.get(i) != labels.get(j):
                    continue
                a_ij = max(adjacency.get(i, {}).get(j, 0.0), 0.0)
                q += a_ij - degree[i] * degree[j] / total
        return q / total

    # ------------------------------------------------------------------
    # 社群演化：关系强度随时间动态变化 + 未来状态预测
    # ------------------------------------------------------------------
    def social_evolution(self, steps: int = 1, dt: float = 1.0) -> Dict:
        growth = self.config.get("growth_rate", 0.1)
        decay_rate = self.config.get("decay_rate", 0.05)
        snapshots = []
        for _ in range(steps):
            self._time += dt
            detection = self.community_detection()
            member_of = {pid: label
                         for label, members in detection["communities"].items()
                         for pid in members}
            for (a, b), rels in self.relations.items():
                same_community = member_of.get(a) == member_of.get(b)
                for rel_type in list(rels):
                    strength = rels[rel_type]
                    if rel_type == "enemy":
                        # 敌对关系随时间缓慢冷却
                        rels[rel_type] = _clamp(strength - decay_rate * dt)
                    elif same_community:
                        # 同社群的正向关系逐渐增强
                        rels[rel_type] = _clamp(strength + (1.0 - strength)
                                                * growth * dt)
                    else:
                        # 跨社群的正向关系逐渐疏远
                        rels[rel_type] = _clamp(strength - decay_rate * dt)
            self._graph = None
            snapshot = {
                "time": self._time,
                "num_communities": len(detection["communities"]),
                "modularity": detection["modularity"],
                "avg_strength": self._average_strength(),
            }
            self._history.append(snapshot)
            snapshots.append(snapshot)
        return {"snapshots": snapshots, "prediction": self.predict_evolution(steps)}

    def _average_strength(self) -> float:
        strengths = [s for rels in self.relations.values() for s in rels.values()]
        return sum(strengths) / len(strengths) if strengths else 0.0

    def predict_evolution(self, steps: int = 1) -> Dict:
        """按当前演化趋势外推，预测社群未来状态。"""
        growth = self.config.get("growth_rate", 0.1)
        decay_rate = self.config.get("decay_rate", 0.05)
        detection = self.community_detection()
        member_of = {pid: label
                     for label, members in detection["communities"].items()
                     for pid in members}

        predicted_relations = []
        for (a, b), rels in sorted(self.relations.items()):
            same = member_of.get(a) == member_of.get(b)
            predicted = {}
            for rel_type, strength in rels.items():
                if rel_type == "enemy":
                    future = strength - decay_rate * steps
                elif same:
                    future = strength + (1.0 - strength) * growth * steps
                else:
                    future = strength - decay_rate * steps
                predicted[rel_type] = round(_clamp(future), 6)
            predicted_relations.append({
                "source": a, "target": b, "predicted_types": predicted,
            })

        current_avg = self._average_strength()
        if len(self._history) >= 2:
            trend = self._history[-1]["avg_strength"] - self._history[-2]["avg_strength"]
        else:
            trend = 0.0
        return {
            "steps": steps,
            "predicted_num_communities": len(detection["communities"]),
            "predicted_avg_strength": round(_clamp(current_avg + trend * steps), 6),
            "predicted_relations": predicted_relations,
        }

    # ------------------------------------------------------------------
    # 推荐：协同过滤 + 冷启动
    # ------------------------------------------------------------------
    def recommendation(self, player_id: str, top_k: int = 5) -> Dict:
        adjacency = self._adjacency()
        neighbors = {n: w for n, w in adjacency.get(player_id, {}).items()
                     if w > 0}

        if neighbors:
            friends = self._collaborative_filtering(player_id, adjacency,
                                                    neighbors, top_k)
            strategy = "collaborative_filtering"
        else:
            friends = self._cold_start(player_id, adjacency, top_k)
            strategy = "cold_start"

        communities = self._recommend_communities(player_id, friends, neighbors,
                                                  top_k)
        return {
            "player": player_id,
            "strategy": strategy,
            "friends": friends,
            "communities": communities,
        }

    def _collaborative_filtering(self, player_id, adjacency, neighbors, top_k):
        # 基于共同好友的余弦相似度：朋友的朋友得分最高
        target_norm = math.sqrt(sum(w * w for w in neighbors.values()))
        scores: Dict[str, float] = {}
        for candidate in self.players:
            if candidate == player_id or candidate in neighbors:
                continue
            candidate_neighbors = {n: w for n, w
                                   in adjacency.get(candidate, {}).items()
                                   if w > 0}
            common = set(neighbors) & set(candidate_neighbors)
            if not common:
                continue
            dot = sum(neighbors[n] * candidate_neighbors[n] for n in common)
            norm = math.sqrt(sum(w * w for w in candidate_neighbors.values()))
            if target_norm and norm:
                scores[candidate] = dot / (target_norm * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{"player": pid, "score": round(score, 6)}
                for pid, score in ranked[:top_k]]

    def _cold_start(self, player_id, adjacency, top_k):
        # 新玩家无社交关系：按公会、共同兴趣、等级相近、人气打分
        attrs = self.players.get(player_id, {})
        interests = set(attrs.get("interests", []))
        scores: Dict[str, float] = {}
        for other, other_attrs in self.players.items():
            if other == player_id:
                continue
            score = 0.0
            if attrs.get("guild") and attrs.get("guild") == other_attrs.get("guild"):
                score += 2.0
            shared = interests & set(other_attrs.get("interests", []))
            score += len(shared)
            level_diff = abs(attrs.get("level", 0) - other_attrs.get("level", 0))
            score += 1.0 / (1.0 + level_diff)
            score += 0.1 * len(adjacency.get(other, {}))  # 人气分
            if score > 0:
                scores[other] = score
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{"player": pid, "score": round(score, 6)}
                for pid, score in ranked[:top_k]]

    def _recommend_communities(self, player_id, friends, neighbors, top_k):
        detection = self.community_detection()
        friend_ids = {f["player"] for f in friends}
        attrs = self.players.get(player_id, {})
        adjacency = self._adjacency()
        scores = []
        for label, members in detection["communities"].items():
            if player_id in members:
                # 已深度融入（>=2 条社群内正向关系）的社群不再推荐
                internal = sum(1 for m in members
                               if adjacency.get(player_id, {}).get(m, 0) > 0)
                if internal >= 2:
                    continue
            score = (len(friend_ids & set(members))
                     + len(set(neighbors) & set(members)))
            guilds = {self.players.get(m, {}).get("guild") for m in members}
            if attrs.get("guild") and attrs.get("guild") in guilds:
                score += 1.0
            if score > 0:
                scores.append({"community": label, "size": len(members),
                               "score": score})
        scores.sort(key=lambda c: (-c["score"], c["community"]))
        return scores[:top_k]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        self.social_evolution(steps=1, dt=dt)

    def reset(self):
        self.players = {}
        self.relations = {}
        self._graph = None
        self._state = {}
        self._history = []
        self._time = 0.0
