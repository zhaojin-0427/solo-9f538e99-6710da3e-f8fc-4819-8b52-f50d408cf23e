"""分段拼接领域逻辑单元测试。"""
from __future__ import annotations

import itertools
import random

import pytest

from app import stitch
from app.core import Cue, format_timestamp


def vtt(cue_spec):
    """(段内起, 段内止, 文本) -> WebVTT 原文。"""
    out = ["WEBVTT", ""]
    for start_ms, end_ms, text in cue_spec:
        out += [
            f"{format_timestamp(start_ms)} --> {format_timestamp(end_ms)}",
            text,
            "",
        ]
    return "\n".join(out)


def parse(order, cue_spec, start_ms, end_ms, filename=None):
    return stitch.parse_segment(
        order,
        filename or f"seg-{order + 1}.vtt",
        vtt(cue_spec),
        start_ms,
        end_ms,
    )


class TestSegmentParsing:
    def test_local_timecodes_converted_to_original(self):
        seg = parse(0, [(0, 1000, "甲"), (2000, 3000, "乙")], 5000, 9000)
        assert [(c.start_ms, c.end_ms) for c in seg.cues] == [
            (5000, 6000),
            (7000, 8000),
        ]

    def test_out_of_bounds_rejected(self):
        with pytest.raises(stitch.StitchError, match="越界"):
            parse(0, [(0, 5000, "太长")], 0, 4000)

    def test_overlapping_cues_rejected(self):
        with pytest.raises(stitch.StitchError, match="重叠"):
            parse(0, [(0, 2000, "甲"), (1000, 3000, "乙")], 0, 9000)

    def test_unordered_cues_rejected(self):
        with pytest.raises(stitch.StitchError, match="排列"):
            parse(0, [(3000, 4000, "乙"), (0, 1000, "甲")], 0, 9000)

    def test_touching_cues_allowed(self):
        seg = parse(0, [(0, 1000, "甲"), (1000, 2000, "乙")], 0, 9000)
        assert len(seg.cues) == 2

    def test_tag_rejected(self):
        with pytest.raises(stitch.StitchError, match="纯文本"):
            parse(0, [(0, 1000, "带<i>标签</i>")], 0, 9000)

    def test_blank_text_rejected(self):
        with pytest.raises(stitch.StitchError, match="纯文本"):
            parse(0, [(0, 1000, "   ")], 0, 9000)


class TestRangeValidation:
    def test_strictly_increasing_and_overlap(self):
        stitch.validate_ranges([(0, 100), (50, 200), (150, 300)])

    def test_start_must_increase(self):
        with pytest.raises(stitch.StitchError, match="起点"):
            stitch.validate_ranges([(0, 100), (0, 200)])

    def test_end_must_increase(self):
        with pytest.raises(stitch.StitchError, match="终点"):
            stitch.validate_ranges([(0, 100), (50, 100)])

    def test_adjacency_must_overlap(self):
        with pytest.raises(stitch.StitchError, match="重叠"):
            stitch.validate_ranges([(0, 100), (100, 200)])

    def test_count_outside_range(self):
        uploads = [
            {"filename": "a.vtt", "vtt_raw": vtt([(0, 1000, "甲")]),
             "start_ms": 0, "end_ms": 2000}
        ]
        with pytest.raises(stitch.StitchError, match="2～12"):
            stitch.build_segments(uploads)


class TestCandidates:
    def test_candidate_must_be_fully_inside_overlap(self):
        left = parse(0, [(1000, 2000, "早"), (5000, 6000, "同文")], 0, 8000)
        right = parse(1, [(0, 1000, "同文"), (2000, 3000, "后")], 5000, 12000)
        candidates = stitch.find_candidates(left, right, 0)
        assert len(candidates) == 1
        assert candidates[0].text == "同文"
        assert candidates[0].cost == 0

    def test_cue_half_outside_overlap_ignored(self):
        # 前段同文字幕起点早于重叠区起点（right.start_ms=5000）
        left = parse(0, [(4000, 6000, "同文")], 0, 8000)
        right = parse(1, [(0, 2000, "同文")], 5000, 12000)
        assert stitch.find_candidates(left, right, 0) == []

    def test_tolerance_boundary_inclusive(self):
        left = parse(0, [(5000, 6000, "同文")], 0, 8000)
        right = parse(1, [(250, 1250, "同文")], 5000, 12000)  # 偏差各 250
        candidates = stitch.find_candidates(left, right, 0)
        assert len(candidates) == 1
        assert candidates[0].start_delta == -250
        assert candidates[0].end_delta == -250

    def test_beyond_tolerance_rejected(self):
        left = parse(0, [(5000, 6000, "同文")], 0, 8000)
        right = parse(1, [(251, 1251, "同文")], 5000, 12000)
        assert stitch.find_candidates(left, right, 0) == []

    def test_different_text_ignored(self):
        left = parse(0, [(5000, 6000, "甲")], 0, 8000)
        right = parse(1, [(0, 1000, "乙")], 5000, 12000)
        assert stitch.find_candidates(left, right, 0) == []


class TestStitchBasic:
    def _pair(self, left_spec, right_spec, left_range=(0, 8000), right_range=(5000, 12000)):
        left = parse(0, left_spec, *left_range)
        right = parse(1, right_spec, *right_range)
        return stitch.stitch([left, right])

    def test_keeps_left_handoff_and_continues_after_right_match(self):
        result = self._pair(
            [(1000, 2000, "A"), (5000, 6000, "B"), (7000, 7500, "C")],
            [(0, 1000, "B"), (1500, 2500, "D"), (5000, 6000, "E")],
        )
        assert result.ok
        assert [c.text for c in result.stitched] == ["A", "B", "D", "E"]
        # 交接字幕采用前段时码
        handoff = result.stitched[1]
        assert (handoff.start_ms, handoff.end_ms) == (5000, 6000)

    def test_minimises_total_absolute_deviation(self):
        # 重叠区内两条同文候选：近的偏差小，远的偏差大，必须选近的
        result = self._pair(
            [
                (1000, 2000, "A"),
                (5000, 5500, "X"),
                (6000, 6500, "Y"),
            ],
            [
                (100, 600, "X"),    # 原片 5100-5600，与前段 X 偏差各 100
                (1000, 1500, "Y"),  # 原片 6000-6500，偏差 0
            ],
        )
        assert result.ok
        assert [c.text for c in result.plan] == ["Y"]
        assert result.total_cost == 0

    def test_no_solution_marks_both_segments(self):
        result = self._pair(
            [(1000, 2000, "A")],
            [(0, 1000, "完全不同的文字")],
        )
        assert not result.ok
        assert result.problem_segments == [1, 2]

    def test_tie_marks_segments_and_keeps_no_plan(self):
        result = self._pair(
            [(5000, 5500, "X"), (6000, 6500, "X")],
            [(0, 500, "X"), (1000, 1500, "X"), (4000, 5000, "尾")],
            left_range=(0, 10000),
        )
        assert not result.ok
        assert result.problem_segments == [1, 2]
        assert result.plan == [] and result.stitched == []

    def test_stitched_cues_are_sorted_and_non_overlapping(self):
        result = self._pair(
            [(1000, 2000, "A"), (5000, 6000, "B")],
            [(0, 1000, "B"), (1000, 2000, "C"), (4000, 5000, "D")],
        )
        assert result.ok
        times = [(c.start_ms, c.end_ms) for c in result.stitched]
        assert times == sorted(times)
        for (_, end), (start, _) in zip(times, times[1:]):
            assert start >= end

    def test_vtt_text_roundtrips(self):
        result = self._pair(
            [(1000, 2000, "A"), (5000, 6000, "B")],
            [(0, 1000, "B"), (4000, 5000, "D")],
        )
        from app.core import parse_vtt

        reparsed = parse_vtt(result.vtt_text)
        assert [c.text for c in reparsed] == ["A", "B", "D"]
        assert reparsed[1].start_ms == 5000


class TestThreeSegments:
    def _three(self):
        s1 = parse(
            0,
            [(1000, 2000, "A"), (4000, 5000, "H1"), (7000, 8000, "H2")],
            0,
            9000,
        )
        s2 = parse(
            1,
            [(0, 1000, "H1"), (3000, 4000, "H2"), (6000, 7000, "M")],
            4000,
            13000,
        )
        s3 = parse(
            2,
            [(0, 1000, "H2"), (3000, 4000, "Z")],
            7000,
            16000,
        )
        return s1, s2, s3

    def test_full_span_join(self):
        result = stitch.stitch(list(self._three()))
        assert result.ok
        # 前段保留交接字幕：H1 取自第 1 段，H2 取自第 2 段；第 3 段从 H2 之后接起
        assert [c.text for c in result.stitched] == ["A", "H1", "H2", "Z"]
        assert [c.text for c in result.plan] == ["H1", "H2"]
        # 交接字幕分别采用前段记录的原片时码
        h1 = next(c for c in result.stitched if c.text == "H1")
        assert (h1.start_ms, h1.end_ms) == (4000, 5000)
        h2 = next(c for c in result.stitched if c.text == "H2")
        assert (h2.start_ms, h2.end_ms) == (7000, 8000)

    def test_middle_retained_cues_kept_between_handoffs(self):
        s1, _, _ = self._three()
        s2 = parse(
            1,
            [(0, 1000, "H1"), (1500, 2500, "M"), (3000, 4000, "H2")],
            4000,
            13000,
        )
        s3 = parse(2, [(0, 1000, "H2"), (3000, 4000, "Z")], 7000, 16000)
        result = stitch.stitch([s1, s2, s3])
        assert result.ok
        assert [c.text for c in result.stitched] == ["A", "H1", "M", "H2", "Z"]

    def test_prefix_predecessor_chooses_earlier_incoming_index(self):
        # 回归：中间段接入下标 0、交出下标 1，前驱只能来自严格更早的状态。
        s1, s2, s3 = self._three()
        result = stitch.stitch([s1, s2, s3])
        assert result.ok
        # 第一交接：前段 H1 下标 1，中间段接入下标 0
        assert result.plan[0].left_index == 1
        assert result.plan[0].right_index == 0
        # 第二交接：中间段交出下标 1，严格晚于其接入下标 0
        assert result.plan[1].left_index == 1
        assert result.plan[0].right_index < result.plan[1].left_index

    def test_incoming_must_precede_outgoing_in_middle(self):
        s1, s2, s3 = self._three()
        # 让中间段唯一可接入的字幕排在交出字幕之后：交换 H1/H2 次序位置
        s2_bad = parse(
            1,
            [(0, 1000, "H2"), (3000, 4000, "H1"), (6000, 7000, "M")],
            4000,
            13000,
        )
        result = stitch.stitch([s1, s2_bad, s3])
        assert not result.ok
        # 问题应落在涉及中间段的交接上
        assert 2 in result.problem_segments

    def test_broken_middle_adjacency_marked(self):
        s1, _, s3 = self._three()
        s2 = parse(
            1,
            [(0, 1000, "H1"), (6000, 7000, "M")],  # 缺少与第三段交接的 H2
            4000,
            13000,
        )
        result = stitch.stitch([s1, s2, s3])
        assert not result.ok
        assert result.problem_segments == [2, 3]


def _chain(n):
    """构造 n 段、每段 9 秒、步进 4 秒的规则链路，交接均无偏差。"""
    segs = []
    for i in range(n):
        specs = []
        if i > 0:
            specs.append((1000, 1800, f"H{i}"))
        specs.append((2500, 3000, f"only{i}"))
        if i < n - 1:
            specs.append((5000, 5800, f"H{i + 1}"))
        specs = sorted(specs, key=lambda item: item[0])
        segs.append(parse(i, specs, 4000 * i, 4000 * i + 9000, f"s{i}.vtt"))
    return segs


class TestLongChains:
    def test_four_segment_chain_joins_every_adjacency(self):
        result = stitch.stitch(_chain(4))
        assert result.ok
        assert [c.text for c in result.plan] == ["H1", "H2", "H3"]
        assert result.total_cost == 0
        assert [c.text for c in result.stitched] == [
            "only0", "H1", "only1", "H2", "only2", "H3", "only3",
        ]

    def test_twelve_segment_limit_accepted(self):
        result = stitch.stitch(_chain(12))
        assert result.ok
        assert len(result.plan) == 11

    def test_dead_middle_adjacency_localized(self):
        segs = _chain(4)
        # 第 3 段去掉交出字幕，第 3/4 段之间的交接断开
        segs[2] = parse(2, [(1000, 1800, "H2"), (2500, 3000, "only2")], 8000, 17000)
        result = stitch.stitch(segs)
        assert not result.ok
        assert result.problem_segments == [3, 4]

    def test_tie_localized_to_specific_adjacency(self):
        segs = _chain(3)
        # 第 3/4 段重叠区内出现两条零偏差同文候选 => 该交接最优并列
        segs[2] = parse(
            2,
            [(1000, 1800, "H2"), (2500, 3000, "only2"),
             (5000, 5400, "X"), (5600, 6000, "X")],
            8000,
            17000,
        )
        segs.append(
            parse(
                3,
                [(1000, 1400, "X"), (1600, 2000, "X"), (3000, 3500, "only3")],
                12000,
                21000,
            )
        )
        result = stitch.stitch(segs)
        assert not result.ok
        assert result.problem_segments == [3, 4]


class TestBoundaryOverlap:
    def test_cross_boundary_reversal_blocks_plan(self):
        # 前段交接字幕终点 7000；后段接入之后的首条字幕（6500-6800）在段内有序，
        # 但换算到原片后起点 6500 早于 7000，跨段边界倒序，方案须被拒绝。
        left = parse(0, [(5000, 7000, "交接")], 0, 9000)
        right_bad = parse(
            1,
            [(0, 1000, "其他"), (1000, 2000, "交接"), (2100, 2400, "越界")],
            5000,
            12000,
        )
        result = stitch.stitch([left, right_bad])
        assert not result.ok
        assert result.problem_segments == [1, 2]

    def test_touching_boundary_allowed(self):
        # 后段首条保留字幕起点恰好等于前段交接字幕终点，不算重叠。
        left = parse(0, [(5000, 7000, "交接")], 0, 9000)
        right = parse(
            1,
            [(0, 2000, "交接"), (2000, 3000, "恰好相接")],
            5000,
            12000,
        )
        result = stitch.stitch([left, right])
        assert result.ok
        assert [c.text for c in result.stitched] == ["交接", "恰好相接"]


# ---------------------------------------------------------------------------
# 动态规划与暴力枚举一致性
# ---------------------------------------------------------------------------


def _brute_force(segments):
    """枚举所有交接组合，返回 None（无解）/('tie', cost)/('unique', cost)。"""
    by_adjacency, _ = stitch._candidate_lookup(segments)
    if any(not candidates for candidates in by_adjacency):
        return None
    best_cost = None
    plan_count = 0
    for combo in itertools.product(*by_adjacency):
        valid = True
        for i in range(1, len(combo)):
            if combo[i - 1].right_index >= combo[i].left_index:
                valid = False
                break
        if not valid:
            continue
        for cand in combo:
            right = segments[cand.right_segment]
            nxt = cand.right_index + 1
            if nxt < len(right.cues) and right.cues[nxt].start_ms < cand.left_end:
                valid = False
                break
        if not valid:
            continue
        cost = sum(cand.cost for cand in combo)
        if best_cost is None or cost < best_cost:
            best_cost, plan_count = cost, 1
        elif cost == best_cost:
            plan_count += 1
    if best_cost is None:
        return None
    return ("tie" if plan_count > 1 else "unique", best_cost)


def _random_segments(rng, n):
    segments = []
    cursor = 0
    for i in range(n):
        start = cursor
        end = start + rng.randint(8000, 12000)
        cursor = start + rng.randint(3000, end - start - 1000)
        specs = []
        t = 500
        idx = 0
        while t < end - start - 1500:
            dur = rng.randint(500, 1200)
            specs.append((t, t + dur, f"cue{i}-{idx}"))
            t += dur + rng.randint(200, 1500)
            idx += 1
        if i < n - 1:
            overlap_local = cursor - start
            pos = max(overlap_local + 200, (specs[-1][1] + 300) if specs else 0)
            if pos + 800 < end - start:
                specs.append((pos, pos + 800, f"SHARED-{i}"))
                specs.sort(key=lambda item: item[0])
        cues = [Cue(j, "", s, e, t) for j, (s, e, t) in enumerate(specs)]
        segments.append(stitch.Segment(i, f"s{i}.vtt", start, end, cues))
    for i in range(n - 1):
        shared = next((c for c in segments[i].cues if c.text == f"SHARED-{i}"), None)
        if shared is None:
            continue
        shift = rng.choice([0, 0, 0, 100, -150, 250, 400])
        local_start = shared.start_ms - segments[i + 1].start_ms + shift
        local_end = shared.end_ms - segments[i + 1].start_ms + shift
        if local_start < 0 or local_end > segments[i + 1].duration_ms:
            continue
        segments[i + 1].cues.append(
            Cue(len(segments[i + 1].cues), "", local_start, local_end, shared.text)
        )
        segments[i + 1].cues.sort(key=lambda cue: cue.start_ms)
        for j, cue in enumerate(segments[i + 1].cues):
            segments[i + 1].cues[j] = Cue(
                j, "", cue.start_ms, cue.end_ms, cue.text
            )
    return segments


class TestDynamicProgrammingMatchesBruteForce:
    @pytest.mark.parametrize("seed", [1, 7, 42, 99, 12345])
    def test_random_chains(self, seed):
        rng = random.Random(seed)
        for _ in range(60):
            segments = _random_segments(rng, rng.randint(2, 5))
            result = stitch.stitch(segments)
            expected = _brute_force(segments)
            if expected is None:
                assert not result.ok
                continue
            kind, cost = expected
            if kind == "tie":
                assert not result.ok
                continue
            assert result.ok
            assert result.total_cost == cost
            for previous, current in zip(result.stitched, result.stitched[1:]):
                assert current.start_ms >= previous.end_ms
