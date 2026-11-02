import csv, re
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE = Path(r"C:\Users\yaswa\OneDrive\BTP\Decision_Models\eval\outputs\analysis")
OUT = BASE / "matrices_jev_luna_anyjev_l0_semif.xlsx"
MODELS = ["jev", "luna", "anyjev_l0", "semif"]
LABELS = {"jev": "Jev", "luna": "Luna", "anyjev_l0": "AnyJev L0", "semif": "SemIf"}
N_COLS = len(MODELS) + 1

METRIC_ORDER = [
    ("agreement", "Agreement (higher = closer)"),
    ("kappa", "Cohen's kappa (higher = closer)"),
    ("spearman_intra", "Spearman intra-instance (higher = closer)"),
    ("pearson_intra", "Pearson intra-instance (higher = closer)"),
    ("spearman_cross", "Spearman cross-instance on confidence (higher = closer)"),
    ("pearson_cross", "Pearson cross-instance on confidence (higher = closer)"),
    ("tvd", "Total variation distance (lower = closer)"),
    ("jsd", "Jensen-Shannon divergence, bits (lower = closer)"),
    ("kl", "KL(Mi || Mj), bits - directional (lower = closer)"),
]
HIGHER_BETTER = {
    "agreement", "kappa", "spearman_intra", "pearson_intra", "spearman_cross", "pearson_cross"
}

thin_blue = Border(
    left=Side(style="thin", color="1F4E79"),
    right=Side(style="thin", color="1F4E79"),
    top=Side(style="thin", color="1F4E79"),
    bottom=Side(style="thin", color="1F4E79"),
)
thin = Border(
    left=Side(style="thin", color="B0B8C4"),
    right=Side(style="thin", color="B0B8C4"),
    top=Side(style="thin", color="B0B8C4"),
    bottom=Side(style="thin", color="B0B8C4"),
)
fill_title = PatternFill("solid", fgColor="1F4E79")
fill_subtitle = PatternFill("solid", fgColor="2E75B6")
fill_header = PatternFill("solid", fgColor="D6E3F0")
fill_corner = PatternFill("solid", fgColor="1F4E79")
fill_diag = PatternFill("solid", fgColor="E7E6E6")
fill_note = PatternFill("solid", fgColor="FFF2CC")
fill_warn = PatternFill("solid", fgColor="FCE4D6")
fill_section = PatternFill("solid", fgColor="D9EAD3")
fill_mcn_header = PatternFill("solid", fgColor="5B9BD5")
fill_sig = PatternFill("solid", fgColor="F4CCCC")
fill_ns = PatternFill("solid", fgColor="D9EAD3")
font_title = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
font_sub = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
font_h = Font(name="Calibri", size=11, bold=True, color="1F4E79")
font_corner = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
font_cell = Font(name="Calibri", size=11, color="203040")
font_diag = Font(name="Calibri", size=11, bold=True, color="7F7F7F")
font_note = Font(name="Calibri", size=10, italic=True, color="595959")
font_warn = Font(name="Calibri", size=10, color="843C0C")
font_metric = Font(name="Calibri", size=12, bold=True, color="1F4E79")
font_mcn_h = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
font_h2 = Font(name="Calibri", size=12, bold=True, color="1F4E79")
font_warn_h = Font(name="Calibri", size=12, bold=True, color="843C0C")
center = Alignment(horizontal="center", vertical="center", wrap_text=True)
left = Alignment(horizontal="left", vertical="center", wrap_text=True)


def heat_fill(val, metric):
    if metric in HIGHER_BETTER:
        if val >= 0.85:
            return PatternFill("solid", fgColor="C6EFCE")
        if val >= 0.65:
            return PatternFill("solid", fgColor="FFEB9C")
        return PatternFill("solid", fgColor="FFC7CE")
    if metric == "kl":
        if val <= 0.5:
            return PatternFill("solid", fgColor="C6EFCE")
        if val <= 1.5:
            return PatternFill("solid", fgColor="FFEB9C")
        return PatternFill("solid", fgColor="FFC7CE")
    if val <= 0.10:
        return PatternFill("solid", fgColor="C6EFCE")
    if val <= 0.25:
        return PatternFill("solid", fgColor="FFEB9C")
    return PatternFill("solid", fgColor="FFC7CE")


def load_matrix(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header = rows[0][1:]
    mat = {mi: {} for mi in header}
    for row in rows[1:]:
        mi = row[0]
        for mj, cell in zip(header, row[1:]):
            if mi == mj or cell in ("—", "-", ""):
                continue
            try:
                mat[mi][mj] = float(cell)
            except ValueError:
                mat[mi][mj] = cell
    return header, mat


def parse_mcnemar(s):
    m = re.match(r"b=(\d+);\s*c=(\d+);\s*p=([0-9.eE+-]+)\s*\(([^)]+)\)", s.strip())
    if not m:
        return None
    b, c, p, test = m.groups()
    return int(b), int(c), float(p), test.replace("chi2_cc", "chi2 continuity")


def mcnemar_rows(path, models):
    _, mat = load_matrix(path)
    out = []
    for mi in models:
        for mj in models:
            if mi == mj:
                continue
            cell = mat.get(mi, {}).get(mj)
            if not isinstance(cell, str):
                continue
            parsed = parse_mcnemar(cell)
            if not parsed:
                continue
            b, c, p, test = parsed
            out.append(
                (f"{LABELS[mi]} vs {LABELS[mj]}", LABELS[mi], LABELS[mj], b, c, p, test)
            )
    return out


def write_matrix(ws, start_row, metric_key, metric_title, mat, models):
    ws.merge_cells(
        start_row=start_row, start_column=1, end_row=start_row, end_column=N_COLS
    )
    c = ws.cell(start_row, 1, metric_title)
    c.font = font_metric
    c.alignment = left
    c.fill = fill_section
    ws.row_dimensions[start_row].height = 20
    r0 = start_row + 1
    corner = ws.cell(r0, 1, "Mi \\ Mj")
    corner.font = font_corner
    corner.fill = fill_corner
    corner.alignment = center
    corner.border = thin_blue
    for j, m in enumerate(models):
        cell = ws.cell(r0, j + 2, LABELS[m])
        cell.font = font_h
        cell.fill = fill_header
        cell.alignment = center
        cell.border = thin_blue
    for i, mi in enumerate(models):
        row = r0 + 1 + i
        lab = ws.cell(row, 1, LABELS[mi])
        lab.font = font_h
        lab.fill = fill_header
        lab.alignment = center
        lab.border = thin_blue
        ws.row_dimensions[row].height = 22
        for j, mj in enumerate(models):
            cell = ws.cell(row, j + 2)
            cell.alignment = center
            cell.border = thin_blue
            if mi == mj:
                cell.value = "—"
                cell.fill = fill_diag
                cell.font = font_diag
            else:
                v = mat[mi][mj]
                cell.value = round(float(v), 4)
                cell.number_format = "0.0000"
                cell.fill = heat_fill(float(v), metric_key)
                cell.font = font_cell
    ws.row_dimensions[r0].height = 22
    return start_row + 2 + len(models) + 1


def write_mcnemar(ws, start_row, rows):
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=7)
    c = ws.cell(
        start_row,
        1,
        "McNemar on item correctness  (b = Mi right and Mj wrong;  c = Mi wrong and Mj right)",
    )
    c.font = font_metric
    c.fill = fill_section
    c.alignment = left
    headers = ["Pair", "Mi", "Mj", "b", "c", "p-value", "Test"]
    r = start_row + 1
    for j, h in enumerate(headers, 1):
        cell = ws.cell(r, j, h)
        cell.font = font_mcn_h
        cell.fill = fill_mcn_header
        cell.alignment = center
        cell.border = thin
    for i, (pair, mi, mj, b, cval, p, test) in enumerate(rows):
        rr = r + 1 + i
        for j, v in enumerate([pair, mi, mj, b, cval, p, test], 1):
            cell = ws.cell(rr, j, v)
            cell.border = thin
            cell.alignment = center if j > 1 else left
            cell.font = font_cell
            if j == 6:
                cell.number_format = "0.00E+00" if p < 0.001 else "0.0000"
                cell.fill = fill_sig if p < 0.05 else fill_ns
        ws.row_dimensions[rr].height = 20
    return r + 1 + len(rows) + 2


NOTES = {
    "jevbench": "n_units=335 for all pairs  |  gold items=285  |  0 SemIf / AnyJev errors",
    "opendecisions": "jev-AnyJev/SemIf n=2352  |  pairs with Luna n=2346 (6 Luna errors/refusals)",
    "fastdecisions": "jev-Luna: n_units=725, gold=255.  Pairs with AnyJev L0 or SemIf: n_units=711, gold=241 (14 support_intent_* skipped by both).",
}

wb = Workbook()
ws0 = wb.active
ws0.title = "Overview"
ws0.sheet_view.showGridLines = False
ws0.merge_cells("A1:G1")
ws0["A1"] = "BTP — Pairwise Decision-Model Matrices"
ws0["A1"].font = font_title
ws0["A1"].fill = fill_title
ws0["A1"].alignment = Alignment(horizontal="left", vertical="center")
ws0.row_dimensions[1].height = 32
ws0.merge_cells("A2:G2")
ws0["A2"] = (
    "Models: Jev  ·  Luna  ·  AnyJev L0 (Qwen3-4B)  ·  SemIf (Qwen3.5-4B direct)"
    "     |     Generated 27 Oct 2026 (Asia/Calcutta)"
)
ws0["A2"].font = font_sub
ws0["A2"].fill = fill_subtitle
ws0.row_dimensions[2].height = 22
ws0.merge_cells("A4:G4")
ws0["A4"] = "Contents"
ws0["A4"].font = font_h2
ws0.merge_cells("A5:G5")
ws0["A5"] = (
    "One sheet per dataset. Each metric is a 4x4 block (Jev, Luna, AnyJev L0, SemIf). Diagonal = —."
)
ws0["A5"].font = font_note
ws0["A5"].fill = fill_note
ws0["A5"].alignment = left
ws0.row_dimensions[5].height = 32
ws0.merge_cells("A7:G7")
ws0["A7"] = "Metrics included"
ws0["A7"].font = font_h2
ws0.merge_cells("A8:G8")
ws0["A8"] = (
    "Agreement, Cohen's kappa, Spearman/Pearson intra, Spearman/Pearson cross (confidence),"
    " TVD, JSD, KL(Mi||Mj), and McNemar (b, c, p, test)."
)
ws0["A8"].font = font_note
ws0["A8"].fill = fill_note
ws0["A8"].alignment = left
ws0.row_dimensions[8].height = 32
ws0.merge_cells("A10:G10")
ws0["A10"] = "Colour key (value cells)"
ws0["A10"].font = font_h2
ws0.merge_cells("A11:G11")
ws0["A11"] = (
    "Green = closer models  ·  Amber = moderate  ·  Red = farther."
    " For agreement/kappa/correlations, higher is closer; for TVD/JSD/KL, lower is closer."
    " McNemar p < 0.05 highlighted in red."
)
ws0["A11"].font = font_note
ws0["A11"].fill = fill_note
ws0["A11"].alignment = left
ws0.row_dimensions[11].height = 36
for col, lab, color in [
    ("A13", "Closer", "C6EFCE"),
    ("B13", "Moderate", "FFEB9C"),
    ("C13", "Farther", "FFC7CE"),
]:
    ws0[col] = lab
    ws0[col].fill = PatternFill("solid", fgColor=color)
    ws0[col].font = font_cell
    ws0[col].alignment = center
    ws0[col].border = thin

ws0.merge_cells("A15:G15")
ws0["A15"] = "AnyJev L0 and SemIf — Fast Decisions unsupported items (important)"
ws0["A15"].font = font_warn_h
warn_lines = [
    "14 Fast Decisions items failed for both AnyJev L0 and SemIf (support_intent_* with 28 options). AnyJev: letter-readout >26; SemIf: unsupported_option_count:28.",
    "Rows remain in raw JSONL with error set. Pairwise matrices / McNemar for pairs including either model use only items both models answered (n_units 711 / gold 241 vs Jev-Luna 725 / 255).",
    "JevBench and OpenDecisions: 0 SemIf errors. Luna on OpenDecisions has 6 error/refusal rows; Luna pairs there use n=2346.",
    "SemIf accuracy (gold answered): JB 90.2%, OD 67.9%, FD 39.0% (241 graded). Agreement with Jev: JB 0.907, OD 0.746, FD 0.862.",
]
for i, line in enumerate(warn_lines):
    r = 16 + i
    ws0.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
    cell = ws0.cell(r, 1, line)
    cell.font = font_warn
    cell.fill = fill_warn
    cell.alignment = left
    ws0.row_dimensions[r].height = 40

r = 21
ws0.cell(r, 1, "Excluded Fast Decisions item_ids (AnyJev L0 and SemIf)").font = font_h2
err_ids = [
    "fastdecisions_support_intent_0000",
    "fastdecisions_support_intent_0007",
    "fastdecisions_support_intent_0023",
    "fastdecisions_support_intent_0029",
    "fastdecisions_support_intent_0035",
    "fastdecisions_support_intent_0048",
    "fastdecisions_support_intent_0051",
    "fastdecisions_support_intent_0063",
    "fastdecisions_support_intent_0067",
    "fastdecisions_support_intent_0071",
    "fastdecisions_support_intent_0073",
    "fastdecisions_support_intent_0081",
    "fastdecisions_support_intent_0087",
    "fastdecisions_support_intent_0095",
]
ws0.cell(r + 1, 1, "item_id").font = font_mcn_h
ws0.cell(r + 1, 1).fill = fill_mcn_header
ws0.cell(r + 1, 1).border = thin
ws0.cell(r + 1, 2, "note").font = font_mcn_h
ws0.cell(r + 1, 2).fill = fill_mcn_header
ws0.cell(r + 1, 2).border = thin
for i, iid in enumerate(err_ids):
    ws0.cell(r + 2 + i, 1, iid).font = font_cell
    ws0.cell(r + 2 + i, 1).border = thin
    ws0.cell(r + 2 + i, 2, "28 options - skipped by AnyJev L0 and SemIf").font = font_cell
    ws0.cell(r + 2 + i, 2).border = thin
ws0.column_dimensions["A"].width = 44
ws0.column_dimensions["B"].width = 58
for col in range(3, 8):
    ws0.column_dimensions[get_column_letter(col)].width = 14

wsA = wb.create_sheet("Accuracy")
wsA.sheet_view.showGridLines = False
wsA.merge_cells("A1:F1")
wsA["A1"] = "Per-model accuracy (answered gold items)"
wsA["A1"].font = font_title
wsA["A1"].fill = fill_title
acc = list(csv.DictReader(open(BASE / "accuracy.csv", encoding="utf-8-sig")))
headers = list(acc[0].keys())
for j, h in enumerate(headers, 1):
    cell = wsA.cell(3, j, h)
    cell.font = font_mcn_h
    cell.fill = fill_mcn_header
    cell.border = thin
    cell.alignment = center
for i, row in enumerate(acc):
    for j, h in enumerate(headers, 1):
        cell = wsA.cell(4 + i, j, row[h])
        cell.border = thin
        cell.font = font_cell
        cell.alignment = center
for col in range(1, 7):
    wsA.column_dimensions[get_column_letter(col)].width = 18

SHEET_META = [
    ("JevBench", "jevbench", "JevBench (org test)"),
    ("OpenDecisions", "opendecisions", "OpenDecisions (full)"),
    ("Fast Decisions", "fastdecisions", "Fast Decisions (test)"),
]
for sheet_name, key, title in SHEET_META:
    ws = wb.create_sheet(sheet_name)
    ws.sheet_view.showGridLines = False
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=N_COLS)
    ws["A1"] = f"BTP pairwise matrices — {title}"
    ws["A1"].font = font_title
    ws["A1"].fill = fill_title
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=N_COLS)
    ws["A2"] = (
        "Models on axes: Jev · Luna · AnyJev L0 · SemIf    |    "
        "Cell (Mi, Mj) = metric between row model and column model"
    )
    ws["A2"].font = font_sub
    ws["A2"].fill = fill_subtitle
    ws.row_dimensions[2].height = 20
    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=N_COLS)
    ws["A3"] = NOTES[key]
    ws["A3"].font = font_note
    ws["A3"].fill = fill_note
    ws["A3"].alignment = left
    ws.row_dimensions[3].height = 34
    row = 5
    for mkey, mtitle in METRIC_ORDER:
        _, mat = load_matrix(BASE / key / f"matrix_{mkey}.csv")
        row = write_matrix(ws, row, mkey, mtitle, mat, MODELS)
    row = write_mcnemar(ws, row, mcnemar_rows(BASE / key / "matrix_mcnemar.csv", MODELS))
    ws.column_dimensions["A"].width = 38
    for col in range(2, 9):
        ws.column_dimensions[get_column_letter(col)].width = 14
    ws.freeze_panes = "A4"

wb.save(OUT)
print("Wrote", OUT)
print("size", OUT.stat().st_size)
