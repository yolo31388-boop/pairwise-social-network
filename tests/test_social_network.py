"""
游戏内玩家社群与社交网络系统 - 验收测试
运行方式：python -m pytest tests/test_social_network.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from social.network import SocialNetwork


def build_two_cliques(system):
    """两个内部紧密、之间只有一条弱边的小团体。"""
    for a, b, s in [("a1", "a2", 0.9), ("a2", "a3", 0.9), ("a1", "a3", 0.8),
                    ("b1", "b2", 0.9), ("b2", "b3", 0.9), ("b1", "b3", 0.8),
                    ("a3", "b1", 0.1)]:
        system.add_relation(a, b, "friend", s)


class TestSocialNetwork:
    def setup_method(self):
        self.system = SocialNetwork()

    # ---- 社交网络构建：关系类型（好友/师徒/夫妻/敌人） ----
    def test_case_01(self):
        self.system.add_relation("p1", "p2", "friend", 0.8)
        self.system.add_relation("p1", "p3", "mentor", 0.6)
        self.system.add_relation("p2", "p3", "spouse", 0.9)
        self.system.add_relation("p3", "p4", "enemy", 0.7)
        graph = self.system.build_graph()
        all_types = set()
        for edge in graph["edges"]:
            all_types.update(edge["types"].keys())
        assert {"friend", "mentor", "spouse", "enemy"} <= all_types

    # ---- 社交网络构建：图结构完整 ----
    def test_case_02(self):
        self.system.add_relation("p1", "p2", "friend", 0.8)
        self.system.add_relation("p2", "p3", "friend", 0.5)
        graph = self.system.build_graph()
        assert {n["id"] for n in graph["nodes"]} == {"p1", "p2", "p3"}
        assert len(graph["edges"]) == 2
        assert graph["adjacency"]["p1"]["p2"] == graph["adjacency"]["p2"]["p1"]

    # ---- 社交网络构建：关系强度决定边权重 ----
    def test_case_03(self):
        self.system.add_relation("p1", "p2", "friend", 0.9)
        self.system.add_relation("p1", "p3", "friend", 0.2)
        self.system.add_relation("p1", "p4", "enemy", 0.8)
        graph = self.system.build_graph()
        adj = graph["adjacency"]
        assert adj["p1"]["p2"] > adj["p1"]["p3"]
        assert adj["p1"]["p4"] < 0  # 敌对关系为负权重

    # ---- 影响力传播：热门话题能从种子扩散 ----
    def test_case_04(self):
        self.system.add_relation("p1", "p2", "friend", 0.9)
        self.system.add_relation("p2", "p3", "friend", 0.9)
        result = self.system.influence_propagation("新版本上线", ["p1"])
        assert result["model"] == "weighted_cascade"
        assert "p2" in result["influenced"]
        assert "p3" in result["influenced"]

    # ---- 影响力传播：距离越远衰减越明显 ----
    def test_case_05(self):
        self.system.add_relation("p1", "p2", "friend", 1.0)
        self.system.add_relation("p2", "p3", "friend", 1.0)
        self.system.add_relation("p3", "p4", "friend", 1.0)
        result = self.system.influence_propagation("赛事", ["p1"], decay=0.5)
        inf = result["influenced"]
        assert inf["p2"] > inf["p3"] > inf["p4"]

    # ---- 影响力传播：阈值过滤 + 敌对关系不传播 ----
    def test_case_06(self):
        self.system.add_relation("p1", "p2", "friend", 0.1)
        self.system.add_relation("p1", "p3", "enemy", 0.9)
        result = self.system.influence_propagation("话题", ["p1"],
                                                   threshold=0.5)
        assert "p2" not in result["influenced"]  # 衰减后低于阈值
        assert "p3" not in result["influenced"]  # 敌对关系阻断传播

    # ---- 社群发现：能发现玩家小团体 ----
    def test_case_07(self):
        build_two_cliques(self.system)
        result = self.system.community_detection()
        communities = [set(m) for m in result["communities"].values()]
        assert result["algorithm"] == "label_propagation"
        assert {"a1", "a2", "a3"} in communities
        assert {"b1", "b2", "b3"} in communities

    # ---- 社群发现：模块度评估社群质量 ----
    def test_case_08(self):
        build_two_cliques(self.system)
        result = self.system.community_detection()
        assert -1.0 <= result["modularity"] <= 1.0
        assert result["modularity"] > 0.3  # 明显的小团体结构应有较高模块度

    # ---- 社群演化：关系随时间动态变化 ----
    def test_case_09(self):
        self.system.add_relation("p1", "p2", "friend", 0.5)
        before = self.system.relations[("p1", "p2")]["friend"]
        result = self.system.social_evolution(steps=3)
        after = self.system.relations[("p1", "p2")]["friend"]
        assert before != after
        assert len(result["snapshots"]) == 3
        assert result["snapshots"][-1]["time"] > result["snapshots"][0]["time"]

    # ---- 社群演化：预测未来发展 ----
    def test_case_10(self):
        build_two_cliques(self.system)
        result = self.system.social_evolution(steps=2)
        prediction = result["prediction"]
        assert prediction["predicted_num_communities"] >= 1
        assert len(prediction["predicted_relations"]) == 7
        for rel in prediction["predicted_relations"]:
            for strength in rel["predicted_types"].values():
                assert 0.0 <= strength <= 1.0

    # ---- 推荐：协同过滤推荐好友（朋友的朋友） ----
    def test_case_11(self):
        # p1 与 p2/p3 是好友，p2/p3 都与 p4 是好友 -> 应推荐 p4 给 p1
        self.system.add_relation("p1", "p2", "friend", 0.9)
        self.system.add_relation("p1", "p3", "friend", 0.8)
        self.system.add_relation("p2", "p4", "friend", 0.9)
        self.system.add_relation("p3", "p4", "friend", 0.9)
        result = self.system.recommendation("p1")
        assert result["strategy"] == "collaborative_filtering"
        recommended = [f["player"] for f in result["friends"]]
        assert "p4" in recommended
        assert "p2" not in recommended  # 已是好友不再推荐

    # ---- 推荐：新玩家冷启动 ----
    def test_case_12(self):
        self.system.add_player("new", guild="星辰", level=10,
                               interests=["pvp", "raid"])
        self.system.add_player("v1", guild="星辰", level=12,
                               interests=["pvp"])
        self.system.add_player("v2", guild="暗夜", level=50,
                               interests=["休闲"])
        result = self.system.recommendation("new")
        assert result["strategy"] == "cold_start"
        assert result["friends"][0]["player"] == "v1"  # 同公会+同兴趣优先

    # ---- 推荐：推荐合适的社群 ----
    def test_case_13(self):
        build_two_cliques(self.system)
        self.system.add_relation("me", "a1", "friend", 0.9)
        result = self.system.recommendation("me")
        assert result["communities"]
        best = result["communities"][0]
        assert best["score"] > 0
        # 被推荐的社群应包含好友 a1 所在的整个小团体
        members = best["community"]
        assert members  # label 为该社群成员之一
        assert best["size"] >= 3

    # ---- 生命周期：update 推进演化，reset 清空状态 ----
    def test_case_14(self):
        self.system.add_relation("p1", "p2", "friend", 0.5)
        self.system.update(1.0)
        assert len(self.system._history) == 1
        self.system.reset()
        assert self.system.players == {}
        assert self.system.relations == {}
        assert self.system._history == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
