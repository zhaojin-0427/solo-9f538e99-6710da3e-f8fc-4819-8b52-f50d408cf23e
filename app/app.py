"""口述史馆检索字幕 —— Flask 网页入口。"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from . import core, stitch
from .stitch import MAX_SEGMENTS, MIN_SEGMENTS

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "oral_history.db")


def create_app(database_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        DATABASE=os.environ.get("ORAL_HISTORY_DB")
        or database_path
        or DEFAULT_DB,
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,
        SECRET_KEY=os.environ.get("ORAL_HISTORY_SECRET", "oral-history-local"),
        JSON_AS_ASCII=False,
    )
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    core.init_db(app.config["DATABASE"])
    app.teardown_appcontext(close_db)

    @app.template_filter("ts")
    def _ts(value: int) -> str:
        return core.format_timestamp(value)

    @app.get("/")
    def index():
        conn = get_db()
        draft = core.active_draft(conn)
        draft_info = None
        if draft is not None:
            bundle = core.load_bundle(conn, draft["id"])
            draft_info = {
                "id": draft["id"],
                "revision": draft["revision"],
                "vtt_filename": draft["vtt_filename"],
                "csv_filename": draft["csv_filename"],
                "created_at": draft["created_at"],
                "fragment_count": len(bundle["cuts"]),
                "cue_count": len(bundle["cues"]),
                "occurrence_count": len(bundle["occurrences"]),
                "cut_count": sum(
                    1 for occ in bundle["occurrences"] if occ.kind == core.CUT
                ),
                "pending_count": len(
                    core.pending_occurrences(bundle["fragments"], bundle["decisions"])
                ),
                "total_ms": sum(f.duration_ms for f in bundle["fragments"]),
            }
        return render_template(
            "index.html", draft=draft_info, editions=core.list_editions(conn)
        )

    @app.post("/stitch")
    def stitch_upload():
        uploads, csv_raw, csv_name, error = _read_segment_form()
        if error is not None:
            flash(error, "error")
            return redirect(url_for("index"))
        try:
            cuts = core.parse_cuts_csv(csv_raw)
        except core.ParseError as exc:
            flash(f"剪辑 CSV 无法使用：{exc}", "error")
            return redirect(url_for("index"))
        try:
            segments = stitch.build_segments(uploads)
            result = stitch.stitch(segments)
        except stitch.StitchError as exc:
            flash(f"分段拼接失败：{exc}", "error")
            return redirect(url_for("index"))

        if not result.ok:
            # 无解或最优并列：标出有问题的分段，保留当前草稿，不建新草稿。
            return render_template(
                "stitch_problem.html",
                result=result,
                csv_name=csv_name,
            ), 422

        payload = _preview_payload(result, cuts)
        conn = get_db()
        try:
            preview_id = core.save_stitch_preview(
                conn,
                result.vtt_text,
                cuts,
                payload,
                segment_filenames=[item["filename"] for item in uploads],
                csv_filename=csv_name,
            )
        except Exception:  # 写入失败：保留现有草稿。
            app.logger.exception("拼接预览写入失败")
            flash("拼接预览写入失败，现有草稿已保留，请重试。", "error")
            return redirect(url_for("index"))
        return redirect(url_for("stitch_preview_page", preview_id=preview_id))

    @app.get("/stitch/preview/<int:preview_id>")
    def stitch_preview_page(preview_id: int):
        conn = get_db()
        row = core.load_stitch_preview(conn, preview_id)
        if row is None:
            abort(404)
        payload = json.loads(row["payload_json"])
        stale = row["lock_version"] != core.lock_version(conn)
        return render_template(
            "stitch_preview.html",
            preview=row,
            payload=payload,
            stale=stale,
        ), (409 if stale else 200)

    @app.post("/stitch/preview/<int:preview_id>/confirm")
    def stitch_confirm(preview_id: int):
        conn = get_db()
        row = core.load_stitch_preview(conn, preview_id)
        if row is None:
            abort(404)
        try:
            draft_id = core.confirm_stitch_preview(conn, preview_id)
        except core.StalePreview as exc:
            payload = json.loads(row["payload_json"])
            return (
                render_template(
                    "stitch_preview.html",
                    preview=row,
                    payload=payload,
                    stale=True,
                    stale_message=str(exc),
                ),
                409,
            )
        except Exception:  # 写入失败：保留现有草稿。
            app.logger.exception("拼接确认写入失败")
            flash("新草稿写入失败，现有草稿已保留，请重试。", "error")
            return redirect(url_for("index"))
        flash("已按合成字幕与剪辑表建立新草稿，请重新校订剪口。", "ok")
        return redirect(url_for("draft_page", draft_id=draft_id))

    @app.get("/draft/<int:draft_id>")
    def draft_page(draft_id: int):
        conn = get_db()
        bundle = core.load_bundle(conn, draft_id)
        if bundle is None:
            abort(404)
        view = _draft_view(conn, bundle)
        return render_template("draft.html", **view), (409 if view["stale"] else 200)

    @app.post("/draft/<int:draft_id>/save")
    def save(draft_id: int):
        conn = get_db()
        bundle = core.load_bundle(conn, draft_id)
        if bundle is None:
            abort(404)
        if bundle["stale"]:
            return _stale_response(conn, draft_id)
        revision = _form_revision()
        changes = _collect_changes(bundle, request.form)
        try:
            core.save_decisions(conn, draft_id, revision, changes)
        except core.StaleDraft:
            return _stale_response(conn, draft_id)
        except core.ValidationError as exc:
            flash(str(exc), "error")
            return redirect(url_for("draft_page", draft_id=draft_id))
        flash("校订已保存。", "ok")
        return redirect(url_for("draft_page", draft_id=draft_id))

    @app.post("/draft/<int:draft_id>/publish")
    def publish(draft_id: int):
        conn = get_db()
        bundle = core.load_bundle(conn, draft_id)
        if bundle is None:
            abort(404)
        if bundle["stale"]:
            return _stale_response(conn, draft_id)
        revision = _form_revision()
        changes = _collect_changes(bundle, request.form)
        try:
            # 页面上的编辑与发布一次提交：先按同一版本号保存，再发布。
            if changes:
                core.save_decisions(conn, draft_id, revision, changes)
            version_no = core.publish(conn, draft_id, revision)
        except core.StaleDraft:
            return _stale_response(conn, draft_id)
        except core.ValidationError as exc:
            flash(str(exc), "error")
            return redirect(url_for("draft_page", draft_id=draft_id))
        except core.PublishBlocked as exc:
            bundle = core.load_bundle(conn, draft_id)
            view = _draft_view(conn, bundle)
            view["error"] = (
                f"还有 {len(exc.pending)} 处剪口字幕未处理，处理完才能发布。"
                "已处理的内容已保留。"
            )
            return render_template("draft.html", **view), 422
        flash(f"已发布第 {version_no} 版。", "ok")
        return redirect(url_for("edition_detail", version_no=version_no))

    @app.get("/editions")
    def editions():
        conn = get_db()
        return render_template("editions.html", editions=core.list_editions(conn))

    @app.get("/editions/<int:version_no>")
    def edition_detail(version_no: int):
        conn = get_db()
        edition = core.get_edition(conn, version_no)
        if edition is None:
            abort(404)
        body = edition["vtt_blob"].decode("utf-8")
        return render_template("edition.html", edition=edition, body=body)

    @app.get("/editions/<int:version_no>/download")
    def download(version_no: int):
        conn = get_db()
        edition = core.get_edition(conn, version_no)
        if edition is None:
            abort(404)
        return send_file(
            io.BytesIO(edition["vtt_blob"]),
            mimetype="text/vtt",
            as_attachment=True,
            download_name=f"oral-history-v{version_no}.vtt",
            max_age=0,
        )

    return app


# ---------------------------------------------------------------------------
# 请求辅助
# ---------------------------------------------------------------------------


def get_db():
    from flask import current_app

    if "db" not in g:
        g.db = core.connect(current_app.config["DATABASE"])
    return g.db


def close_db(_exc=None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def _form_revision() -> int:
    try:
        return int(request.form.get("revision", ""))
    except (TypeError, ValueError):
        abort(400, "缺少校订版本号")


def _decode_upload(file_storage, label: str) -> str:
    try:
        return file_storage.read().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise stitch.StitchError(f"{label}不是 UTF-8 编码") from exc


def _read_segment_form() -> tuple[list[dict], str, str, Optional[str]]:
    """解析首页多段上传表单。

    成功时返回 ``(上传项, CSV 原文, CSV 文件名, None)``；
    失败时最后一项为错误信息。
    """
    count = len(request.files.getlist("vtt"))
    if not MIN_SEGMENTS <= count <= MAX_SEGMENTS:
        return [], "", "", (
            f"请按原片顺序上传 {MIN_SEGMENTS}～{MAX_SEGMENTS} 份转写，"
            f"当前收到 {count} 份文件。"
        )

    vtt_files = request.files.getlist("vtt")
    start_values = request.form.getlist("start_ms")
    end_values = request.form.getlist("end_ms")
    if not (len(start_values) == len(end_values) == count):
        return [], "", "", "每一段都要填写原片起止毫秒，并各选一份 WebVTT。"

    uploads: list[dict] = []
    try:
        for order, vtt_file in enumerate(vtt_files):
            label = f"第 {order + 1} 段"
            if not vtt_file.filename:
                return [], "", "", f"{label}缺少 WebVTT 文件。"
            start_text = start_values[order].strip()
            end_text = end_values[order].strip()
            if not start_text or not end_text:
                return [], "", "", f"{label}缺少原片起止毫秒。"
            uploads.append(
                {
                    "filename": secure_filename(vtt_file.filename)
                    or vtt_file.filename,
                    "vtt_raw": _decode_upload(vtt_file, label),
                    "start_ms": start_text,
                    "end_ms": end_text,
                }
            )
    except stitch.StitchError as exc:
        return [], "", "", str(exc)

    csv_file = request.files.get("csv")
    if csv_file is None or not csv_file.filename:
        return [], "", "", "请选择剪辑 CSV 文件。"
    csv_bytes = csv_file.read()
    try:
        csv_raw = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        # 中文表格软件导出的 CSV 常为 GBK/GB18030。
        try:
            csv_raw = csv_bytes.decode("gb18030")
        except UnicodeDecodeError:
            return [], "", "", "剪辑 CSV 不是 UTF-8 或 GB18030 编码。"
    csv_name = secure_filename(csv_file.filename) or csv_file.filename
    return uploads, csv_raw, csv_name, None


def _preview_payload(result, cuts) -> dict:
    """组织预览页所需数据：各交接采用的字幕 + 合成稿原片时码。"""
    handoffs = []
    for cand in result.plan:
        handoffs.append(
            {
                "adjacency": cand.adjacency,
                "left_segment": cand.left_segment + 1,
                "right_segment": cand.right_segment + 1,
                "text": cand.text,
                "left_start": cand.left_start,
                "left_end": cand.left_end,
                "right_start": cand.right_start,
                "right_end": cand.right_end,
                "start_delta": cand.start_delta,
                "end_delta": cand.end_delta,
                "cost": cand.cost,
            }
        )
    stitched = [
        {
            "start_ms": cue.start_ms,
            "end_ms": cue.end_ms,
            "text": cue.text,
            "segment": cue.segment + 1,
        }
        for cue in result.stitched
    ]
    segments = [
        {
            "order": seg.order + 1,
            "filename": seg.filename,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "cue_count": len(seg.cues),
        }
        for seg in result.segments
    ]
    return {
        "handoffs": handoffs,
        "stitched": stitched,
        "segments": segments,
        "cuts": [cut.__dict__ for cut in cuts],
        "total_cost": result.total_cost,
        "cue_count": len(stitched),
        "vtt_text": result.vtt_text,
    }


def _collect_changes(bundle: dict, form) -> list[dict]:
    """按出现的表单序号收集校订：status:<n> / text:<n>。"""
    changes: list[dict] = []
    for index, occ in enumerate(bundle["occurrences"]):
        if occ.kind != core.CUT:
            continue
        status = form.get(f"status:{index}")
        if status is None:
            continue
        changes.append(
            {
                "signature": occ.signature,
                "status": status,
                "text": form.get(f"text:{index}", ""),
            }
        )
    return changes


def _stale_response(conn, draft_id: int):
    bundle = core.load_bundle(conn, draft_id)
    if bundle is None:
        abort(404)
    view = _draft_view(conn, bundle)
    view["stale"] = True
    view["stale_message"] = "校订页已过期：剪辑表已更换或该页已提交过，请刷新后重新录入。"
    return render_template("draft.html", **view), 409


def _draft_view(conn, bundle: dict) -> dict:
    row = bundle["row"]
    fragments = bundle["fragments"]
    decisions = bundle["decisions"]

    previous = conn.execute(
        "SELECT id FROM draft WHERE id < ? ORDER BY id DESC LIMIT 1", (row["id"],)
    ).fetchone()
    previous_decisions = (
        core.decision_rows(conn, previous["id"]) if previous is not None else {}
    )

    cards = []
    occurrence_index = -1
    for fragment in fragments:
        frag_cards = []
        for occ in fragment.occurrences:
            occurrence_index += 1
            decision = decisions.get(occ.signature)
            frag_cards.append(
                {
                    "index": occurrence_index,
                    "occ": occ,
                    "decision": decision,
                    "inherited": occ.kind == core.CUT
                    and occ.signature in previous_decisions,
                    "pending": core.is_pending(occ, decisions),
                }
            )
        cards.append({"fragment": fragment, "occurrences": frag_cards})

    pending = core.pending_occurrences(fragments, decisions)
    return {
        "row": row,
        "cards": cards,
        "pending_count": len(pending),
        "fragment_count": len(fragments),
        "occurrence_count": len(bundle["occurrences"]),
        "cut_count": sum(1 for occ in bundle["occurrences"] if occ.kind == core.CUT),
        "total_ms": sum(fragment.duration_ms for fragment in fragments),
        "stale": bundle["stale"],
        "stale_message": (
            "校订页已过期：剪辑表已更换或该页已提交过，请刷新后重新录入。"
            if bundle["stale"]
            else None
        ),
        "error": None,
    }
