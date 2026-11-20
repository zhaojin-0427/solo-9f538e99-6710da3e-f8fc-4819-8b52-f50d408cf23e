"""分段转写拼接领域逻辑（不依赖 Flask，便于单独测试）。

访谈按段转写，交回的 WebVTT 使用段内时码（每段从 0 开始）。本模块负责：

* 解析并校验各段的原片起止与段内字幕（纯文本、按序、不重叠、不越界）；
* 在相邻段的重叠区内寻找同文字幕作为交接候选（换算到原片后
  起点、终点偏差各不超过 ``TOLERANCE_MS``）；
* 用动态规划在所有能接通全期的方案中，取各交接起止绝对偏差之和最小的一组；
* 无解或最优并列时标出有问题的分段；
* 按方案拼出原片时码的合成字幕。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .core import Cue, ParseError, build_vtt, parse_timestamp, parse_vtt

# 交接换算到原片后，起点 / 终点各自允许的最大绝对偏差（毫秒）。
TOLERANCE_MS = 250
# 首页允许同时上传的转写份数。
MIN_SEGMENTS = 2
MAX_SEGMENTS = 12

# 纯文本字幕不允许出现 WebVTT 尖括号标签。
_TAG_RE = re.compile(r"<[^>]+>")

# DP 中的不可达状态与并列标记。
_INF = float("inf")
_TIED = -1


class StitchError(ValueError):
    """分段输入或拼接过程不合法。"""


@dataclass(frozen=True)
class Segment:
    """一段转写：原片起止毫秒 + 段内时码字幕（已换算为原片时码）。"""

    order: int
    filename: str
    start_ms: int          # 该段在原片上的起点
    end_ms: int            # 该段在原片上的终点（不含）
    cues: list[Cue] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def label(self) -> str:
        return f"第 {self.order + 1} 段"


@dataclass(frozen=True)
class HandoffCandidate:
    """相邻两段在重叠区内的一条同文交接候选。"""

    adjacency: int          # 交接序号，0 = 第 1/2 段之间
    left_segment: int       # 前段序号（0 基）
    right_segment: int      # 后段序号（0 基）
    left_index: int         # 交接字幕在前段 cues 中的下标
    right_index: int        # 交接字幕在后段 cues 中的下标
    text: str
    left_start: int         # 前段记录的原片起点
    left_end: int
    right_start: int        # 后段记录的原片起点
    right_end: int
    start_delta: int
    end_delta: int

    @property
    def cost(self) -> int:
        """该交接的起止绝对偏差之和。"""
        return abs(self.start_delta) + abs(self.end_delta)


@dataclass(frozen=True)
class StitchedCue:
    """合成稿中的一条字幕（原片时码）。"""

    start_ms: int
    end_ms: int
    text: str
    segment: int


@dataclass
class StitchResult:
    """拼接求解结果。

    ``ok`` 为真时 ``plan`` 给出每个交接采用的候选，``stitched`` 为合成字幕；
    为假时 ``problem_segments`` 标出有问题的分段（从 1 开始的段号）。
    """

    ok: bool
    segments: list[Segment]
    plan: list[HandoffCandidate] = field(default_factory=list)
    stitched: list[StitchedCue] = field(default_factory=list)
    problem_segments: list[int] = field(default_factory=list)
    reason: str = ""

    @property
    def total_cost(self) -> int:
        return sum(cand.cost for cand in self.plan)

    @property
    def vtt_text(self) -> str:
        return build_vtt(
            [(cue.start_ms, cue.end_ms, cue.text.split("\n")) for cue in self.stitched]
        )


# ---------------------------------------------------------------------------
# 分段解析与校验
# ---------------------------------------------------------------------------


def _require_plain_text(text: str, segment_label: str) -> None:
    if not text.strip():
        raise StitchError(f"{segment_label}存在空白字幕，字幕正文须为纯文本")
    if _TAG_RE.search(text):
        raise StitchError(f"{segment_label}字幕含有标签，段内字幕只允许纯文本")


def parse_segment(
    order: int,
    filename: str,
    vtt_raw: str,
    start_ms: int,
    end_ms: int,
) -> Segment:
    """解析一段转写并把段内时码换算成原片时码，同时做段内校验。"""
    label = f"第 {order + 1} 段"
    if end_ms <= start_ms:
        raise StitchError(f"{label}的原片止毫秒须大于起毫秒")
    try:
        local_cues = parse_vtt(vtt_raw)
    except ParseError as exc:
        raise StitchError(f"{label}（{filename}）WebVTT 解析失败：{exc}") from exc

    cues: list[Cue] = []
    previous_end = 0
    for cue in local_cues:
        _require_plain_text(cue.text, label)
        if cue.start_ms < 0 or cue.end_ms > end_ms - start_ms:
            raise StitchError(
                f"{label}字幕「{cue.text[:12]}」越界：段内时码超出该段原片范围"
            )
        if cue.start_ms < previous_end:
            raise StitchError(
                f"{label}字幕未按时码排列或存在重叠：「{cue.text[:12]}」"
            )
        previous_end = cue.end_ms
        cues.append(
            Cue(
                index=len(cues),
                identifier=cue.identifier,
                start_ms=start_ms + cue.start_ms,
                end_ms=start_ms + cue.end_ms,
                text=cue.text,
                settings=cue.settings,
            )
        )
    if not cues:
        raise StitchError(f"{label}（{filename}）中没有字幕")
    return Segment(order, filename, start_ms, end_ms, cues)


def validate_ranges(ranges: list[tuple[int, int]]) -> None:
    """各段起点、终点都须逐段递增，且相邻段要有重叠。"""
    for i, (start_ms, end_ms) in enumerate(ranges):
        label = f"第 {i + 1} 段"
        if end_ms <= start_ms:
            raise StitchError(f"{label}的原片止毫秒须大于起毫秒")
        if i == 0:
            continue
        prev_start, prev_end = ranges[i - 1]
        if start_ms <= prev_start:
            raise StitchError(f"{label}的原片起点须大于前一段起点")
        if end_ms <= prev_end:
            raise StitchError(f"{label}的原片终点须大于前一段终点")
        if start_ms >= prev_end:
            raise StitchError(
                f"第 {i} 段与第 {i + 1} 段没有重叠：相邻段的原片时段必须相交"
            )


def build_segments(uploads: list[dict]) -> list[Segment]:
    """从上传项构造分段列表。

    每个上传项：``{"filename", "vtt_raw", "start_ms", "end_ms"}``，
    顺序即原片顺序。
    """
    if not MIN_SEGMENTS <= len(uploads) <= MAX_SEGMENTS:
        raise StitchError(
            f"请按原片顺序上传 {MIN_SEGMENTS}～{MAX_SEGMENTS} 份转写，"
            f"当前为 {len(uploads)} 份"
        )
    ranges: list[tuple[int, int]] = []
    for item in uploads:
        try:
            start_ms = parse_timestamp(str(item["start_ms"]))
            end_ms = parse_timestamp(str(item["end_ms"]))
        except ParseError as exc:
            raise StitchError(
                f"第 {len(ranges) + 1} 段原片起止毫秒无法识别：{exc}"
            ) from exc
        ranges.append((start_ms, end_ms))
    validate_ranges(ranges)

    segments: list[Segment] = []
    for order, (item, (start_ms, end_ms)) in enumerate(zip(uploads, ranges)):
        segments.append(
            parse_segment(
                order,
                item.get("filename") or f"segment-{order + 1}.vtt",
                item["vtt_raw"],
                start_ms,
                end_ms,
            )
        )
    return segments


# ---------------------------------------------------------------------------
# 交接候选
# ---------------------------------------------------------------------------


def find_candidates(
    left: Segment, right: Segment, adjacency: int
) -> list[HandoffCandidate]:
    """找出相邻两段重叠区内完整出现的同文交接候选。

    同文字幕须在两段中都完整落在重叠区 ``[right.start_ms, left.end_ms)`` 内；
    换算到原片后起点、终点偏差各不超过 :data:`TOLERANCE_MS`。
    """
    overlap_start = right.start_ms
    overlap_end = min(left.end_ms, right.end_ms)
    if overlap_end <= overlap_start:
        return []

    right_by_text: dict[str, list[int]] = {}
    for idx, cue in enumerate(right.cues):
        if cue.start_ms >= overlap_start and cue.end_ms <= overlap_end:
            right_by_text.setdefault(cue.text, []).append(idx)

    candidates: list[HandoffCandidate] = []
    for left_index, cue in enumerate(left.cues):
        if not (cue.start_ms >= overlap_start and cue.end_ms <= overlap_end):
            continue
        for right_index in right_by_text.get(cue.text, []):
            other = right.cues[right_index]
            start_delta = cue.start_ms - other.start_ms
            end_delta = cue.end_ms - other.end_ms
            if (
                abs(start_delta) <= TOLERANCE_MS
                and abs(end_delta) <= TOLERANCE_MS
            ):
                candidates.append(
                    HandoffCandidate(
                        adjacency=adjacency,
                        left_segment=left.order,
                        right_segment=right.order,
                        left_index=left_index,
                        right_index=right_index,
                        text=cue.text,
                        left_start=cue.start_ms,
                        left_end=cue.end_ms,
                        right_start=other.start_ms,
                        right_end=other.end_ms,
                        start_delta=start_delta,
                        end_delta=end_delta,
                    )
                )
    return candidates


def _candidate_lookup(
    segments: list[Segment],
) -> tuple[list[list[HandoffCandidate]], list[dict[int, list[HandoffCandidate]]]]:
    """返回 (每交接的候选列表, 每交接按后段字幕下标分组的候选)。"""
    by_adjacency: list[list[HandoffCandidate]] = []
    grouped: list[dict[int, list[HandoffCandidate]]] = []
    for adjacency in range(len(segments) - 1):
        candidates = find_candidates(
            segments[adjacency], segments[adjacency + 1], adjacency
        )
        by_adjacency.append(candidates)
        groups: dict[int, list[HandoffCandidate]] = {}
        for cand in candidates:
            groups.setdefault(cand.right_index, []).append(cand)
        grouped.append(groups)
    return by_adjacency, grouped


# ---------------------------------------------------------------------------
# 动态规划
#
# 状态按「交接 + 该交接后段接入字幕下标」组织。前向状态记录抵达该状态的
# 最小代价、方案数（>1 截断为 2）与前驱候选；后向状态记录从该状态走到
# 末段的最小代价。转移时强制：
#   1. 中间段接入下标 < 交出下标；
#   2. 前段最后保留字幕与后段第一条保留字幕不倒序、不重叠。
# ---------------------------------------------------------------------------


@dataclass
class _Fwd:
    cost: float
    count: int
    pred: Optional[HandoffCandidate]   # 唯一最优前驱；并列时为 None
    incoming_index: int = -1           # 本状态接入字幕在该段的下标


def _boundary_ok(
    segments: list[Segment],
    cand: HandoffCandidate,
    first_retained_index: Optional[int],
) -> bool:
    """后段第一条保留字幕不得早于前段交接字幕的终点。"""
    if first_retained_index is None:
        return True
    first_cue = segments[cand.right_segment].cues[first_retained_index]
    return first_cue.start_ms >= cand.left_end


def _forward_dp(
    segments: list[Segment],
    grouped: list[dict[int, list[HandoffCandidate]]],
) -> list[dict[int, _Fwd]]:
    n_adj = len(segments) - 1
    forward: list[dict[int, _Fwd]] = []
    for adjacency in range(n_adj):
        right_segment = segments[adjacency + 1]
        current: dict[int, _Fwd] = {}
        for right_index, incoming in grouped[adjacency].items():
            nxt = right_index + 1
            first_retained = nxt if nxt < len(right_segment.cues) else None
            best: Optional[_Fwd] = None
            for cand in incoming:
                if not _boundary_ok(segments, cand, first_retained):
                    continue
                if adjacency == 0:
                    cand_cost: float = cand.cost
                    cand_count = 1
                    cand_pred: Optional[HandoffCandidate] = None
                else:
                    # 中间段接入下标严格小于交出下标，在所有合规前驱中取最优。
                    prev = _prefix_best(
                        forward[adjacency - 1], cand.left_index
                    )
                    if prev is None:
                        continue
                    cand_cost = prev.cost + cand.cost
                    cand_count = prev.count
                    # 前驱状态唯一（第一交接本就没有更深前驱）才能还原整条链。
                    cand_pred = cand if prev.count == 1 else None
                if best is None or cand_cost < best.cost:
                    best = _Fwd(
                        cand_cost, cand_count, cand_pred, incoming_index=right_index
                    )
                elif cand_cost == best.cost:
                    best = _Fwd(
                        cand_cost,
                        min(2, best.count + cand_count),
                        None,
                        incoming_index=right_index,
                    )
            if best is not None:
                current[right_index] = best
        forward.append(current)
    return forward


def _backward_dp(
    segments: list[Segment],
    by_adjacency: list[list[HandoffCandidate]],
) -> list[dict[int, float]]:
    """每个「交接 + 前段交出下标」走到末段的最小代价；不可达不在字典中。"""
    n_adj = len(segments) - 1
    backward: list[dict[int, float]] = [dict() for _ in range(n_adj)]
    for adjacency in range(n_adj - 1, -1, -1):
        current: dict[int, float] = {}
        for cand in by_adjacency[adjacency]:
            right_segment = segments[cand.right_segment]
            nxt = cand.right_index + 1
            first_retained = nxt if nxt < len(right_segment.cues) else None
            if not _boundary_ok(segments, cand, first_retained):
                continue
            if adjacency == n_adj - 1:
                tail: float = cand.cost
            else:
                suffix = backward[adjacency + 1]
                # 下一交接的交出下标必须严格大于本交接在该段的接入下标。
                reachable = [
                    suffix[idx]
                    for idx in suffix
                    if idx > cand.right_index
                ]
                if not reachable:
                    continue
                tail = cand.cost + min(reachable)
            existing = current.get(cand.left_index)
            if existing is None or tail < existing:
                current[cand.left_index] = tail
        backward[adjacency] = current
    return backward


def _prefix_best(
    states: dict[int, "_Fwd"], index: int
) -> Optional["_Fwd"]:
    """前一交接中，接入下标严格小于 ``index`` 的可达最优状态。"""
    candidates = [
        state for idx, state in states.items() if state.incoming_index < index
    ]
    return min(candidates, key=lambda state: state.cost, default=None)


def _suffix_cost(suffix: dict[int, float], index: int) -> float:
    """下一交接中，交出下标严格大于 ``index`` 的最小后缀代价。"""
    reachable = [cost for idx, cost in suffix.items() if idx > index]
    return min(reachable, default=_INF)


def _dead_adjacencies(
    n_adj: int,
    by_adjacency: list[list[HandoffCandidate]],
    forward: list[dict[int, _Fwd]],
    backward: list[dict[int, float]],
) -> list[int]:
    """不属于任何可接通全期方案的交接。"""
    dead: list[int] = []
    for adjacency in range(n_adj):
        alive = False
        for cand in by_adjacency[adjacency]:
            if adjacency == 0:
                before_ok = True
            else:
                before_ok = _prefix_best(
                    forward[adjacency - 1], cand.left_index
                ) is not None
            if adjacency == n_adj - 1:
                after_ok = True
            else:
                after_ok = _suffix_cost(
                    backward[adjacency + 1], cand.right_index
                ) < _INF
            if before_ok and after_ok:
                alive = True
                break
        if not alive:
            dead.append(adjacency)
    return dead


def _segments_for_adjacencies(adjacencies, n_adj: int) -> list[int]:
    """把交接序号换算为涉及的 1 基段号并去重排序。"""
    seen: set[int] = set()
    for adj in adjacencies:
        seen.add(adj + 1)       # 前段
        seen.add(adj + 2)       # 后段
    return sorted(seen)


def _optimal_tied_adjacencies(
    forward: list[dict[int, "_Fwd"]], optimum: float
) -> list[int]:
    """沿最优代价定位发生方案合并（count>1）的交接序号。"""
    tied: list[int] = []
    for adjacency, states in enumerate(forward):
        if any(
            state.cost <= optimum and state.count > 1 for state in states.values()
        ):
            tied.append(adjacency)
    return tied


def _tied_adjacencies(
    n_adj: int,
    by_adjacency: list[list[HandoffCandidate]],
    forward: list[dict[int, _Fwd]],
    backward: list[dict[int, float]],
    optimum: float,
) -> list[int]:
    """最优总代价下，存在两条及以上候选边的交接（即最优并列发生处）。"""
    tied: list[int] = []
    for adjacency in range(n_adj):
        optimal_edges = 0
        for cand in by_adjacency[adjacency]:
            if adjacency == 0:
                before: float = 0
            else:
                prefix = _prefix_best(forward[adjacency - 1], cand.left_index)
                before = prefix.cost if prefix is not None else _INF
            if adjacency == n_adj - 1:
                after: float = 0
            else:
                after = _suffix_cost(backward[adjacency + 1], cand.right_index)
            if before + cand.cost + after == optimum:
                optimal_edges += 1
        if optimal_edges > 1:
            tied.append(adjacency)
    return tied


def _reconstruct(
    forward: list[dict[int, _Fwd]],
    grouped: list[dict[int, list[HandoffCandidate]]],
    optimum_index: int,
) -> list[HandoffCandidate]:
    """从唯一最优末状态沿前驱还原完整方案。"""
    n_adj = len(forward)
    chosen: list[Optional[HandoffCandidate]] = [None] * n_adj
    state_index = optimum_index
    for adjacency in range(n_adj - 1, -1, -1):
        state = forward[adjacency][state_index]
        if adjacency == 0:
            # 第一交接没有前驱：在同下标候选里取构成最优代价的那一条。
            match = next(
                cand
                for cand in grouped[adjacency][state_index]
                if cand.cost == state.cost
            )
            chosen[adjacency] = match
            break
        if state.pred is None:  # pragma: no cover - 调用前已排除并列
            raise StitchError("交接方案还原失败：存在并列前驱")
        chosen[adjacency] = state.pred
        # 回到前一交接的最优前缀状态（接入下标即前驱候选的 right_index）。
        prev_index = state.pred.right_index
        prev_state = forward[adjacency - 1][prev_index]
        if prev_state.cost + state.pred.cost != state.cost:  # pragma: no cover
            raise StitchError("交接方案还原失败：前驱代价不一致")
        state_index = prev_index
    return [cand for cand in chosen if cand is not None]


def stitch(segments: list[Segment]) -> StitchResult:
    """求解分段拼接。"""
    if not MIN_SEGMENTS <= len(segments) <= MAX_SEGMENTS:
        raise StitchError("分段数量超出允许范围")

    by_adjacency, grouped = _candidate_lookup(segments)
    n_adj = len(segments) - 1

    empty_adjacencies = [
        adj for adj, candidates in enumerate(by_adjacency) if not candidates
    ]
    forward = _forward_dp(segments, grouped)
    final_states = forward[-1] if forward else {}

    if not final_states:
        backward = _backward_dp(segments, by_adjacency)
        problem_adj = empty_adjacencies or _dead_adjacencies(
            n_adj, by_adjacency, forward, backward
        )
        return StitchResult(
            ok=False,
            segments=segments,
            reason="重叠区内找不到符合 250 毫秒容差的同文交接字幕，无法接通全期",
            problem_segments=_segments_for_adjacencies(
                problem_adj or range(n_adj), n_adj
            ),
        )

    optimum = min(state.cost for state in final_states.values())
    backward = _backward_dp(segments, by_adjacency)
    tied = _tied_adjacencies(n_adj, by_adjacency, forward, backward, optimum)
    optimal_final_states = [
        idx for idx, state in final_states.items() if state.cost == optimum
    ]
    if (
        tied
        or len(optimal_final_states) > 1
        or any(final_states[idx].count > 1 for idx in optimal_final_states)
    ):
        problem_adj = tied or _optimal_tied_adjacencies(forward, optimum)
        return StitchResult(
            ok=False,
            segments=segments,
            reason="存在多组偏差总和相同的最优交接方案，无法唯一确定，请人工核对标出的分段",
            problem_segments=_segments_for_adjacencies(
                problem_adj or range(n_adj), n_adj
            ),
        )

    optimum_index = next(
        idx for idx, state in final_states.items() if state.cost == optimum
    )
    plan = _reconstruct(forward, grouped, optimum_index)
    stitched = _assemble(segments, plan)
    return StitchResult(
        ok=True,
        segments=segments,
        plan=plan,
        stitched=stitched,
    )


# ---------------------------------------------------------------------------
# 合成
# ---------------------------------------------------------------------------


def _assemble(
    segments: list[Segment], plan: list[HandoffCandidate]
) -> list[StitchedCue]:
    """按方案拼接：保留前段交接字幕，后段从交接字幕的下一条接起。"""
    stitched: list[StitchedCue] = []
    for order, segment in enumerate(segments):
        if order == 0:
            lower = 0
        else:
            lower = plan[order - 1].right_index + 1
        if order < len(segments) - 1:
            upper = plan[order].left_index + 1     # 含交接字幕
        else:
            upper = len(segment.cues)
        for cue in segment.cues[lower:upper]:
            stitched.append(
                StitchedCue(
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    text=cue.text,
                    segment=order,
                )
            )

    # 拼后字幕不得倒序或重叠（段内已保证有序，这里再兜底校验跨段边界）。
    for previous, current in zip(stitched, stitched[1:]):
        if current.start_ms < previous.end_ms:
            raise StitchError(
                "合成字幕在交接处出现倒序或重叠，请检查交接字幕选择"
            )
    return stitched
