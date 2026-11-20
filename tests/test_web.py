"""网页流程测试：多段上传、拼接预览、确认建稿、校订、发布、过期与失败保留。"""
from __future__ import annotations

import io
import re
import sqlite3

import pytest

from app import core
from app.app import create_app

# ---------------------------------------------------------------------------
# 分段夹具（段内时码，每段从 0 开始）
# ---------------------------------------------------------------------------

SEG1 = """WEBVTT

1
00:00:01.000 --> 00:00:02.000
第一句。

2
00:00:05.000 --> 00:00:06.000
交接句。

3
00:00:07.000 --> 00:00:07.500
段一尾。
"""

SEG2 = """WEBVTT

1
00:00:00.000 --> 00:00:01.000
交接句。

2
00:00:02.000 --> 00:00:03.000
段二头。

3
00:00:05.000 --> 00:00:06.000
段二尾。
"""

# 剪辑表作用于拼接后的原片时间轴（0–11000 毫秒）。
CSV = """片段号,原片起毫秒,原片止毫秒
A,1000,5500
B,5500,11000
"""

CSV_NO_CUTS = """片段号,原片起毫秒,原片止毫秒
A,1000,11000
"""


@pytest.fixture()
def app(tmp_path):
    application = create_app(str(tmp_path / "test.db"))
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def raw(text: str):
    return io.BytesIO(text.encode("utf-8"))


def stitch_form(
    client,
    segments=(("a.vtt", SEG1, "0", "8000"), ("b.vtt", SEG2, "5000", "12000")),
    csv_text=CSV,
):
    data = {}
    for name, content, _start, _end in segments:
        data.setdefault("vtt", []).append((raw(content), name))
    data["start_ms"] = [item[2] for item in segments]
    data["end_ms"] = [item[3] for item in segments]
    data["csv"] = (raw(csv_text), "edit.csv")
    return client.post(
        "/stitch",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=False,
    )


def preview_id_of(response) -> int:
    match = re.search(r"/stitch/preview/(\d+)", response.headers["Location"])
    assert match, response.headers["Location"]
    return int(match.group(1))


def draft_id_of(response) -> int:
    match = re.search(r"/draft/(\d+)", response.headers["Location"])
    assert match, response.headers["Location"]
    return int(match.group(1))


class TestIndex:
    def test_homepage_offers_multi_segment_form(self, client):
        html = client.get("/").data.decode()
        assert "2～12" in html
        assert 'name="vtt"' in html
        assert 'name="start_ms"' in html
        assert 'name="end_ms"' in html
        assert 'name="csv"' in html


class TestStitchUpload:
    def test_valid_upload_redirects_to_preview(self, client):
        response = stitch_form(client)
        assert response.status_code == 302
        assert "/stitch/preview/" in response.headers["Location"]

    def test_preview_shows_handoff_and_original_timeline(self, client):
        preview_id = preview_id_of(stitch_form(client))
        page = client.get(f"/stitch/preview/{preview_id}")
        assert page.status_code == 200
        html = page.data.decode()
        # 各处采用的交接字幕
        assert "交接句。" in html
        # 交接的前段原片时码与后段换算时码
        assert "00:00:05.000" in html and "00:00:06.000" in html
        # 合成稿原片时码
        assert "第一句。" in html and "段二头。" in html and "段二尾。" in html
        # 偏差总和为 0
        assert "0 毫秒" in html

    def test_stitched_vtt_drops_duplicate_handoff(self, client):
        preview_id = preview_id_of(stitch_form(client))
        html = client.get(f"/stitch/preview/{preview_id}").data.decode()
        # 交接句只保留一次（前段版本），段一尾因位于重叠区但不交接仍保留
        assert html.count("交接句。") >= 1
        # 确认建稿后合成 VTT 中交接句仅一条
        response = client.post(f"/stitch/preview/{preview_id}/confirm")
        draft_id = draft_id_of(response)
        conn = core.connect(client.application.config["DATABASE"])
        try:
            row = core.load_draft(conn, draft_id)
            cues = core.parse_vtt(row["vtt_raw"])
        finally:
            conn.close()
        texts = [cue.text for cue in cues]
        assert texts == ["第一句。", "交接句。", "段二头。", "段二尾。"]
        handoff = cues[1]
        assert (handoff.start_ms, handoff.end_ms) == (5000, 6000)

    def test_too_few_segments_rejected(self, client):
        follow = client.post(
            "/stitch",
            data={
                "vtt": [(raw(SEG1), "a.vtt")],
                "start_ms": ["0"],
                "end_ms": ["8000"],
                "csv": (raw(CSV), "edit.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert "2～12" in follow.data.decode()

    def test_non_increasing_ranges_rejected(self, client):
        follow = client.post(
            "/stitch",
            data={
                "vtt": [(raw(SEG1), "a.vtt"), (raw(SEG2), "b.vtt")],
                "start_ms": ["5000", "0"],
                "end_ms": ["12000", "8000"],
                "csv": (raw(CSV), "edit.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert "起点" in follow.data.decode()

    def test_non_overlapping_adjacency_rejected(self, client):
        follow = client.post(
            "/stitch",
            data={
                "vtt": [(raw(SEG1), "a.vtt"), (raw(SEG2), "b.vtt")],
                "start_ms": ["0", "8000"],
                "end_ms": ["8000", "12000"],
                "csv": (raw(CSV), "edit.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert "重叠" in follow.data.decode()

    def test_missing_end_ms_rejected(self, client):
        follow = client.post(
            "/stitch",
            data={
                "vtt": [(raw(SEG1), "a.vtt"), (raw(SEG2), "b.vtt")],
                "start_ms": ["0", "5000"],
                "end_ms": ["8000", ""],
                "csv": (raw(CSV), "edit.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert "起止毫秒" in follow.data.decode()

    def test_bad_vtt_rejected(self, client):
        follow = client.post(
            "/stitch",
            data={
                "vtt": [(raw("not vtt"), "a.vtt"), (raw(SEG2), "b.vtt")],
                "start_ms": ["0", "5000"],
                "end_ms": ["8000", "12000"],
                "csv": (raw(CSV), "edit.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert "WebVTT" in follow.data.decode()

    def test_bad_csv_rejected(self, client):
        follow = client.post(
            "/stitch",
            data={
                "vtt": [(raw(SEG1), "a.vtt"), (raw(SEG2), "b.vtt")],
                "start_ms": ["0", "5000"],
                "end_ms": ["8000", "12000"],
                "csv": (raw("片段号,起,止\nA,1000,1000\n"), "edit.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert "剪辑 CSV" in follow.data.decode()

    def test_no_solution_renders_problem_page_and_keeps_draft(self, client):
        # 两段没有任何同文字幕
        other = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n完全不同。\n"
        response = client.post(
            "/stitch",
            data={
                "vtt": [(raw(SEG1), "a.vtt"), (raw(other), "b.vtt")],
                "start_ms": ["0", "5000"],
                "end_ms": ["8000", "12000"],
                "csv": (raw(CSV), "edit.csv"),
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 422
        html = response.data.decode()
        assert "未能接通全期" in html
        assert "有问题" in html
        # 未建立任何草稿
        conn = core.connect(client.application.config["DATABASE"])
        try:
            assert conn.execute("SELECT COUNT(*) FROM draft").fetchone()[0] == 0
        finally:
            conn.close()


class TestConfirmAndDraft:
    def test_confirm_creates_draft_and_opens_review(self, client):
        preview_id = preview_id_of(stitch_form(client))
        response = client.post(f"/stitch/preview/{preview_id}/confirm")
        assert response.status_code == 302
        draft_id = draft_id_of(response)
        page = client.get(f"/draft/{draft_id}")
        assert page.status_code == 200
        html = page.data.decode()
        # 剪口校订页给出原片与节选时码
        assert "原片" in html and "节选" in html

    def test_confirm_with_cut_cues_then_publish(self, client):
        preview_id = preview_id_of(stitch_form(client, csv_text=CSV))
        draft_id = draft_id_of(
            client.post(f"/stitch/preview/{preview_id}/confirm")
        )
        html = client.get(f"/draft/{draft_id}").data.decode()
        fields = sorted({int(m) for m in re.findall(r'name="status:(\d+)"', html)})
        assert fields  # 剪辑表在 5500 处切开了交接句，存在剪口
        revision = re.search(r'name="revision" value="(\d+)"', html).group(1)
        data = {"revision": revision}
        for index in fields:
            data[f"status:{index}"] = "keep"
            data[f"text:{index}"] = f"校订{index}"
        published = client.post(f"/draft/{draft_id}/publish", data=data)
        assert published.status_code == 302
        assert "/editions/1" in published.headers["Location"]
        body = client.get("/editions/1/download").data.decode()
        assert body.startswith("WEBVTT")
        assert "校订" in body

    def test_publish_blocked_until_cuts_handled(self, client):
        preview_id = preview_id_of(stitch_form(client))
        draft_id = draft_id_of(
            client.post(f"/stitch/preview/{preview_id}/confirm")
        )
        html = client.get(f"/draft/{draft_id}").data.decode()
        revision = re.search(r'name="revision" value="(\d+)"', html).group(1)
        response = client.post(
            f"/draft/{draft_id}/publish", data={"revision": revision}
        )
        assert response.status_code == 422
        assert "未处理" in response.data.decode()

    def test_whole_only_csv_publishes_without_decisions(self, client):
        preview_id = preview_id_of(stitch_form(client, csv_text=CSV_NO_CUTS))
        draft_id = draft_id_of(
            client.post(f"/stitch/preview/{preview_id}/confirm")
        )
        html = client.get(f"/draft/{draft_id}").data.decode()
        revision = re.search(r'name="revision" value="(\d+)"', html).group(1)
        response = client.post(
            f"/draft/{draft_id}/publish", data={"revision": revision}
        )
        assert response.status_code == 302
        body = client.get("/editions/1/download").data.decode()
        for text in ("第一句。", "交接句。", "段二头。", "段二尾。"):
            assert text in body


class TestStalePreview:
    def _two_previews(self, client):
        first = preview_id_of(stitch_form(client))
        second = preview_id_of(stitch_form(client))
        return first, second

    def test_confirm_after_newer_preview_prompts_re_preview(self, client):
        first, second = self._two_previews(client)
        # 确认第二个预览会推进锁版本，使第一个预览过期
        assert client.post(f"/stitch/preview/{second}/confirm").status_code == 302
        stale = client.post(f"/stitch/preview/{first}/confirm")
        assert stale.status_code == 409
        assert "重新预览" in stale.data.decode()

    def test_preview_page_flags_stale(self, client):
        first, second = self._two_previews(client)
        client.post(f"/stitch/preview/{second}/confirm")
        page = client.get(f"/stitch/preview/{first}")
        assert page.status_code == 409
        assert "预览已过期" in page.data.decode()

    def test_publish_bumps_lock_and_duplicate_confirm_rejected(self, client):
        preview_id = preview_id_of(stitch_form(client, csv_text=CSV_NO_CUTS))
        draft_id = draft_id_of(
            client.post(f"/stitch/preview/{preview_id}/confirm")
        )
        revision = re.search(
            r'name="revision" value="(\d+)"',
            client.get(f"/draft/{draft_id}").data.decode(),
        ).group(1)
        client.post(f"/draft/{draft_id}/publish", data={"revision": revision})
        # 发布后同一预览再次确认必须被拒绝
        again = client.post(f"/stitch/preview/{preview_id}/confirm")
        assert again.status_code == 409
        assert "重新预览" in again.data.decode()


class TestWriteFailure:
    def test_failed_confirm_keeps_existing_draft(self, client, monkeypatch):
        preview_id = preview_id_of(stitch_form(client, csv_text=CSV_NO_CUTS))
        # 先成功建立一份草稿
        existing_id = draft_id_of(
            client.post(f"/stitch/preview/{preview_id}/confirm")
        )

        def boom(*_args, **_kwargs):
            raise sqlite3.OperationalError("simulated disk failure")

        monkeypatch.setattr(core, "confirm_stitch_preview", boom)
        response = client.post(f"/stitch/preview/{preview_id}/confirm")
        assert response.status_code == 302  # 回到工作台并提示
        follow = client.get(response.headers["Location"], follow_redirects=True)
        assert "写入失败" in follow.data.decode()
        # 现有草稿仍然可用，未被破坏
        conn = core.connect(client.application.config["DATABASE"])
        try:
            assert core.load_draft(conn, existing_id) is not None
            assert conn.execute("SELECT COUNT(*) FROM draft").fetchone()[0] == 1
        finally:
            conn.close()


class TestPersistence:
    def test_edition_survives_restart(self, tmp_path):
        db = str(tmp_path / "restart.db")
        first = create_app(db)
        first.config.update(TESTING=True)
        client = first.test_client()
        preview_id = preview_id_of(stitch_form(client, csv_text=CSV_NO_CUTS))
        draft_id = draft_id_of(
            client.post(f"/stitch/preview/{preview_id}/confirm")
        )
        revision = re.search(
            r'name="revision" value="(\d+)"',
            client.get(f"/draft/{draft_id}").data.decode(),
        ).group(1)
        client.post(f"/draft/{draft_id}/publish", data={"revision": revision})
        expected = client.get("/editions/1/download").data

        second = create_app(db)
        second.config.update(TESTING=True)
        again = second.test_client().get("/editions/1/download")
        assert again.status_code == 200
        assert again.data == expected
