from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from pathlib import Path

OUT = Path(r"C:\Users\yaswa\OneDrive\BTP\Decision_Models\eval\outputs\analysis\matrices_jev_luna_anyjev_l0.xlsx")

MODELS = ["jev", "luna", "anyjev_l0"]
LABELS = {"jev": "Jev", "luna": "Luna", "anyjev_l0": "AnyJev L0"}

DATA = {
  "jevbench": {
    "n_note": "n_units=335  |  k>=3 intra=137  |  gold items=285  |  0 AnyJev errors",
    "agreement": {"jev":{"luna":0.943284,"anyjev_l0":0.850746},"luna":{"jev":0.943284,"anyjev_l0":0.832836},"anyjev_l0":{"jev":0.850746,"luna":0.832836}},
    "kappa": {"jev":{"luna":0.930618,"anyjev_l0":0.817727},"luna":{"jev":0.930618,"anyjev_l0":0.795812},"anyjev_l0":{"jev":0.817727,"luna":0.795812}},
    "spearman_intra": {"jev":{"luna":0.904871,"anyjev_l0":0.711781},"luna":{"jev":0.904871,"anyjev_l0":0.686235},"anyjev_l0":{"jev":0.711781,"luna":0.686235}},
    "pearson_intra": {"jev":{"luna":0.900052,"anyjev_l0":0.678571},"luna":{"jev":0.900052,"anyjev_l0":0.653252},"anyjev_l0":{"jev":0.678571,"luna":0.653252}},
    "spearman_cross": {"jev":{"luna":0.717021,"anyjev_l0":0.501986},"luna":{"jev":0.717021,"anyjev_l0":0.531515},"anyjev_l0":{"jev":0.501986,"luna":0.531515}},
    "pearson_cross": {"jev":{"luna":0.709059,"anyjev_l0":0.118579},"luna":{"jev":0.709059,"anyjev_l0":0.0802953},"anyjev_l0":{"jev":0.118579,"luna":0.0802953}},
    "tvd": {"jev":{"luna":0.0631337,"anyjev_l0":0.159579},"luna":{"jev":0.0631337,"anyjev_l0":0.165804},"anyjev_l0":{"jev":0.159579,"luna":0.165804}},
    "jsd": {"jev":{"luna":0.0268787,"anyjev_l0":0.123273},"luna":{"jev":0.0268787,"anyjev_l0":0.136309},"anyjev_l0":{"jev":0.123273,"luna":0.136309}},
    "kl": {"jev":{"luna":0.202772,"anyjev_l0":2.15985},"luna":{"jev":0.185615,"anyjev_l0":2.27836},"anyjev_l0":{"jev":0.804836,"luna":1.23068}},
    "mcnemar": [
      ("Jev vs Luna", "Jev", "Luna", 8, 3, 0.226562, "exact"),
      ("Luna vs Jev", "Luna", "Jev", 3, 8, 0.226562, "exact"),
      ("Jev vs AnyJev L0", "Jev", "AnyJev L0", 33, 2, 3.95886e-07, "chi2 continuity"),
      ("AnyJev L0 vs Jev", "AnyJev L0", "Jev", 2, 33, 3.95886e-07, "chi2 continuity"),
      ("Luna vs AnyJev L0", "Luna", "AnyJev L0", 32, 6, 5.00196e-05, "chi2 continuity"),
      ("AnyJev L0 vs Luna", "AnyJev L0", "Luna", 6, 32, 5.00196e-05, "chi2 continuity"),
    ],
  },
  "opendecisions": {
    "n_note": "jev-AnyJev n=2352  |  pairs with Luna n=2346 (6 Luna errors/refusals)  |  k>=3 ~1762-1766",
    "agreement": {"jev":{"luna":0.804774,"anyjev_l0":0.656888},"luna":{"jev":0.804774,"anyjev_l0":0.614663},"anyjev_l0":{"jev":0.656888,"luna":0.614663}},
    "kappa": {"jev":{"luna":0.795159,"anyjev_l0":0.638554},"luna":{"jev":0.795159,"anyjev_l0":0.595646},"anyjev_l0":{"jev":0.638554,"luna":0.595646}},
    "spearman_intra": {"jev":{"luna":0.771137,"anyjev_l0":0.560065},"luna":{"jev":0.771137,"anyjev_l0":0.519576},"anyjev_l0":{"jev":0.560065,"luna":0.519576}},
    "pearson_intra": {"jev":{"luna":0.807183,"anyjev_l0":0.544642},"luna":{"jev":0.807183,"anyjev_l0":0.499166},"anyjev_l0":{"jev":0.544642,"luna":0.499166}},
    "spearman_cross": {"jev":{"luna":0.748477,"anyjev_l0":0.429073},"luna":{"jev":0.748477,"anyjev_l0":0.46526},"anyjev_l0":{"jev":0.429073,"luna":0.46526}},
    "pearson_cross": {"jev":{"luna":0.647013,"anyjev_l0":0.19195},"luna":{"jev":0.647013,"anyjev_l0":0.233605},"anyjev_l0":{"jev":0.19195,"luna":0.233605}},
    "tvd": {"jev":{"luna":0.171553,"anyjev_l0":0.341613},"luna":{"jev":0.171553,"anyjev_l0":0.3843},"anyjev_l0":{"jev":0.341613,"luna":0.3843}},
    "jsd": {"jev":{"luna":0.0858411,"anyjev_l0":0.273746},"luna":{"jev":0.0858411,"anyjev_l0":0.306238},"anyjev_l0":{"jev":0.273746,"luna":0.306238}},
    "kl": {"jev":{"luna":0.518497,"anyjev_l0":4.38878},"luna":{"jev":0.764173,"anyjev_l0":5.18865},"anyjev_l0":{"jev":2.71896,"luna":2.47681}},
    "mcnemar": [
      ("Jev vs Luna", "Jev", "Luna", 223, 184, 0.0596204, "chi2 continuity"),
      ("Luna vs Jev", "Luna", "Jev", 184, 223, 0.0596204, "chi2 continuity"),
      ("Jev vs AnyJev L0", "Jev", "AnyJev L0", 624, 93, 3.40621e-87, "chi2 continuity"),
      ("AnyJev L0 vs Jev", "AnyJev L0", "Jev", 93, 624, 3.40621e-87, "chi2 continuity"),
      ("Luna vs AnyJev L0", "Luna", "AnyJev L0", 643, 151, 5.33663e-68, "chi2 continuity"),
      ("AnyJev L0 vs Luna", "AnyJev L0", "Luna", 151, 643, 5.33663e-68, "chi2 continuity"),
    ],
  },
  "fastdecisions": {
    "n_note": "jev-Luna: n_units=725, gold=255.  Pairs with AnyJev L0: n_units=711, gold=241 (14 AnyJev errors excluded).",
    "agreement": {"jev":{"luna":0.878621,"anyjev_l0":0.850914},"luna":{"jev":0.878621,"anyjev_l0":0.850914},"anyjev_l0":{"jev":0.850914,"luna":0.850914}},
    "kappa": {"jev":{"luna":0.850918,"anyjev_l0":0.815222},"luna":{"jev":0.850918,"anyjev_l0":0.81488},"anyjev_l0":{"jev":0.815222,"luna":0.81488}},
    "spearman_intra": {"jev":{"luna":0.827601,"anyjev_l0":0.632279},"luna":{"jev":0.827601,"anyjev_l0":0.629762},"anyjev_l0":{"jev":0.632279,"luna":0.629762}},
    "pearson_intra": {"jev":{"luna":0.876093,"anyjev_l0":0.824544},"luna":{"jev":0.876093,"anyjev_l0":0.789245},"anyjev_l0":{"jev":0.824544,"luna":0.789245}},
    "spearman_cross": {"jev":{"luna":0.715227,"anyjev_l0":0.50712},"luna":{"jev":0.715227,"anyjev_l0":0.466387},"anyjev_l0":{"jev":0.50712,"luna":0.466387}},
    "pearson_cross": {"jev":{"luna":0.574982,"anyjev_l0":0.0905258},"luna":{"jev":0.574982,"anyjev_l0":0.0942351},"anyjev_l0":{"jev":0.0905258,"luna":0.0942351}},
    "tvd": {"jev":{"luna":0.116934,"anyjev_l0":0.162321},"luna":{"jev":0.116934,"anyjev_l0":0.173441},"anyjev_l0":{"jev":0.162321,"luna":0.173441}},
    "jsd": {"jev":{"luna":0.0543564,"anyjev_l0":0.11337},"luna":{"jev":0.0543564,"anyjev_l0":0.124206},"anyjev_l0":{"jev":0.11337,"luna":0.124206}},
    "kl": {"jev":{"luna":0.398224,"anyjev_l0":2.0067},"luna":{"jev":0.383242,"anyjev_l0":2.27094},"anyjev_l0":{"jev":0.704865,"luna":0.747272}},
    "mcnemar": [
      ("Jev vs Luna", "Jev", "Luna", 15, 14, 1.0, "chi2 continuity"),
      ("Luna vs Jev", "Luna", "Jev", 14, 15, 1.0, "chi2 continuity"),
      ("Jev vs AnyJev L0", "Jev", "AnyJev L0", 23, 16, 0.336668, "chi2 continuity"),
      ("AnyJev L0 vs Jev", "AnyJev L0", "Jev", 16, 23, 0.336668, "chi2 continuity"),
      ("Luna vs AnyJev L0", "Luna", "AnyJev L0", 20, 16, 0.617075, "chi2 continuity"),
      ("AnyJev L0 vs Luna", "AnyJev L0", "Luna", 16, 20, 0.617075, "chi2 continuity"),
    ],
  },
}

METRIC_ORDER = [
  ("agreement", "Agreement (higher = closer)"),
  ("kappa", "Cohen's kappa (higher = closer)"),
  ("spearman_intra", "Spearman intra-instance (higher = closer)"),
  ("pearson_intra", "Pearson intra-instance (higher = closer)"),
  ("spearman_cross", "Spearman cross-instance on confidence (higher = closer)"),
  ("pearson_cross", "Pearson cross-instance on confidence (higher = closer)"),
  ("tvd", "Total variation distance (lower = closer)"),
  ("jsd", "Jensen-Shannon divergence, bits (lower = closer)"),
  ("kl", "KL(Mi || Mj), bits — directional (lower = closer)"),
]

HIGHER_BETTER = {"agreement","kappa","spearman_intra","pearson_intra","spearman_cross","pearson_cross"}

thin_blue = Border(
  left=Side(style='thin', color='1F4E79'),
  right=Side(style='thin', color='1F4E79'),
  top=Side(style='thin', color='1F4E79'),
  bottom=Side(style='thin', color='1F4E79'),
)
thin = Border(
  left=Side(style='thin', color='B0B8C4'),
  right=Side(style='thin', color='B0B8C4'),
  top=Side(style='thin', color='B0B8C4'),
  bottom=Side(style='thin', color='B0B8C4'),
)

fill_title = PatternFill('solid', fgColor='1F4E79')
fill_subtitle = PatternFill('solid', fgColor='2E75B6')
fill_header = PatternFill('solid', fgColor='D6E3F0')
fill_corner = PatternFill('solid', fgColor='1F4E79')
fill_diag = PatternFill('solid', fgColor='E7E6E6')
fill_note = PatternFill('solid', fgColor='FFF2CC')
fill_warn = PatternFill('solid', fgColor='FCE4D6')
fill_section = PatternFill('solid', fgColor='D9EAD3')
fill_mcn_header = PatternFill('solid', fgColor='5B9BD5')
fill_sig = PatternFill('solid', fgColor='F4CCCC')
fill_ns = PatternFill('solid', fgColor='D9EAD3')

font_title = Font(name='Calibri', size=16, bold=True, color='FFFFFF')
font_sub = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
font_h = Font(name='Calibri', size=11, bold=True, color='1F4E79')
font_corner = Font(name='Calibri', size=10, bold=True, color='FFFFFF')
font_cell = Font(name='Calibri', size=11, color='203040')
font_diag = Font(name='Calibri', size=11, bold=True, color='7F7F7F')
font_note = Font(name='Calibri', size=10, italic=True, color='595959')
font_warn = Font(name='Calibri', size=10, color='843C0C')
font_metric = Font(name='Calibri', size=12, bold=True, color='1F4E79')
font_mcn_h = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
font_h2 = Font(name='Calibri', size=12, bold=True, color='1F4E79')
font_warn_h = Font(name='Calibri', size=12, bold=True, color='843C0C')

center = Alignment(horizontal='center', vertical='center', wrap_text=True)
left = Alignment(horizontal='left', vertical='center', wrap_text=True)


def heat_fill(val, metric):
    if metric in HIGHER_BETTER:
        if val >= 0.85:
            return PatternFill('solid', fgColor='C6EFCE')
        if val >= 0.65:
            return PatternFill('solid', fgColor='FFEB9C')
        return PatternFill('solid', fgColor='FFC7CE')
    if metric == 'kl':
        if val <= 0.5:
            return PatternFill('solid', fgColor='C6EFCE')
        if val <= 1.5:
            return PatternFill('solid', fgColor='FFEB9C')
        return PatternFill('solid', fgColor='FFC7CE')
    if val <= 0.10:
        return PatternFill('solid', fgColor='C6EFCE')
    if val <= 0.25:
        return PatternFill('solid', fgColor='FFEB9C')
    return PatternFill('solid', fgColor='FFC7CE')


def fmt_num(x, metric):
    return round(float(x), 4)


def write_matrix(ws, start_row, metric_key, metric_title, mat):
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=4)
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
    for j, m in enumerate(MODELS):
        cell = ws.cell(r0, j + 2, LABELS[m])
        cell.font = font_h
        cell.fill = fill_header
        cell.alignment = center
        cell.border = thin_blue
    for i, mi in enumerate(MODELS):
        row = r0 + 1 + i
        lab = ws.cell(row, 1, LABELS[mi])
        lab.font = font_h
        lab.fill = fill_header
        lab.alignment = center
        lab.border = thin_blue
        ws.row_dimensions[row].height = 22
        for j, mj in enumerate(MODELS):
            cell = ws.cell(row, j + 2)
            cell.alignment = center
            cell.border = thin_blue
            if mi == mj:
                cell.value = "—"
                cell.fill = fill_diag
                cell.font = font_diag
            else:
                v = mat[mi][mj]
                cell.value = fmt_num(v, metric_key)
                cell.number_format = '0.0000'
                cell.fill = heat_fill(v, metric_key)
                cell.font = font_cell
    ws.row_dimensions[r0].height = 22
    return start_row + 6


def write_mcnemar(ws, start_row, rows):
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=7)
    c = ws.cell(start_row, 1, "McNemar on item correctness  (b = Mi right & Mj wrong;  c = Mi wrong & Mj right)")
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
        vals = [pair, mi, mj, b, cval, p, test]
        for j, v in enumerate(vals, 1):
            cell = ws.cell(rr, j, v)
            cell.border = thin
            cell.alignment = center if j > 1 else left
            cell.font = font_cell
            if j == 6:
                cell.number_format = '0.00E+00' if p < 0.001 else '0.0000'
                cell.fill = fill_sig if p < 0.05 else fill_ns
        ws.row_dimensions[rr].height = 20
    return r + 1 + len(rows) + 2


wb = Workbook()

# Overview
ws0 = wb.active
ws0.title = "Overview"
ws0.sheet_view.showGridLines = False
ws0.merge_cells('A1:G1')
ws0['A1'] = "BTP — Pairwise Decision-Model Matrices"
ws0['A1'].font = font_title
ws0['A1'].fill = fill_title
ws0['A1'].alignment = Alignment(horizontal='left', vertical='center')
ws0.row_dimensions[1].height = 32

ws0.merge_cells('A2:G2')
ws0['A2'] = "Models: Jev  ·  Luna  ·  AnyJev L0 (Qwen3-4B)     |     Generated 26 Oct 2026 (Asia/Calcutta)"
ws0['A2'].font = font_sub
ws0['A2'].fill = fill_subtitle
ws0.row_dimensions[2].height = 22

ws0.merge_cells('A4:G4')
ws0['A4'] = "Contents"
ws0['A4'].font = font_h2
ws0.merge_cells('A5:G5')
ws0['A5'] = "One sheet per dataset. Each metric is a 4×4 block: corner label, model names on the first row and first column, values in the cells. Diagonal = —."
ws0['A5'].font = font_note
ws0['A5'].fill = fill_note
ws0['A5'].alignment = left
ws0.row_dimensions[5].height = 32

ws0.merge_cells('A7:G7')
ws0['A7'] = "Metrics included"
ws0['A7'].font = font_h2
ws0.merge_cells('A8:G8')
ws0['A8'] = "Agreement, Cohen's kappa, Spearman/Pearson intra, Spearman/Pearson cross (confidence), TVD, JSD, KL(Mi||Mj), and McNemar (b, c, p, test)."
ws0['A8'].font = font_note
ws0['A8'].fill = fill_note
ws0['A8'].alignment = left
ws0.row_dimensions[8].height = 32

ws0.merge_cells('A10:G10')
ws0['A10'] = "Colour key (value cells)"
ws0['A10'].font = font_h2
ws0.merge_cells('A11:G11')
ws0['A11'] = "Green = closer models  ·  Amber = moderate  ·  Red = farther. For agreement/kappa/correlations, higher is closer; for TVD/JSD/KL, lower is closer. McNemar p < 0.05 highlighted in red."
ws0['A11'].font = font_note
ws0['A11'].fill = fill_note
ws0['A11'].alignment = left
ws0.row_dimensions[11].height = 36

# colour legend swatches
ws0['A13'] = "Closer"
ws0['A13'].fill = PatternFill('solid', fgColor='C6EFCE')
ws0['A13'].font = font_cell
ws0['A13'].alignment = center
ws0['A13'].border = thin
ws0['B13'] = "Moderate"
ws0['B13'].fill = PatternFill('solid', fgColor='FFEB9C')
ws0['B13'].font = font_cell
ws0['B13'].alignment = center
ws0['B13'].border = thin
ws0['C13'] = "Farther"
ws0['C13'].fill = PatternFill('solid', fgColor='FFC7CE')
ws0['C13'].font = font_cell
ws0['C13'].alignment = center
ws0['C13'].border = thin

ws0.merge_cells('A15:G15')
ws0['A15'] = "AnyJev L0 — Fast Decisions errors (important)"
ws0['A15'].font = font_warn_h

warn_lines = [
    "14 Fast Decisions items failed for AnyJev L0 with: choice supports at most 26 options in letter readout (all support_intent_* items).",
    "Those rows remain in the raw JSONL with error set and chosen/probs null — they were not dropped from storage.",
    "How metrics treated them: accuracy uses answered gold items only (96/241). For every pairwise matrix and McNemar pair that includes AnyJev L0, only items/units both models answered are used — so these 14 items are excluded from those pairs (n_units 711 / gold 241 vs Jev–Luna 725 / 255).",
    "JevBench and OpenDecisions: 0 AnyJev errors. Luna on OpenDecisions has 6 error/refusal rows; Luna pairs there use n=2346.",
]
for i, line in enumerate(warn_lines):
    r = 16 + i
    ws0.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
    cell = ws0.cell(r, 1, line)
    cell.font = font_warn
    cell.fill = fill_warn
    cell.alignment = left
    ws0.row_dimensions[r].height = 38

r = 21
ws0.cell(r, 1, "Excluded Fast Decisions item_ids (AnyJev L0)").font = font_h2
err_ids = [
    "fastdecisions_support_intent_0000", "fastdecisions_support_intent_0007", "fastdecisions_support_intent_0023",
    "fastdecisions_support_intent_0029", "fastdecisions_support_intent_0035", "fastdecisions_support_intent_0048",
    "fastdecisions_support_intent_0051", "fastdecisions_support_intent_0063", "fastdecisions_support_intent_0067",
    "fastdecisions_support_intent_0071", "fastdecisions_support_intent_0073", "fastdecisions_support_intent_0081",
    "fastdecisions_support_intent_0087", "fastdecisions_support_intent_0095",
]
ws0.cell(r + 1, 1, "item_id").font = font_mcn_h
ws0.cell(r + 1, 1).fill = fill_mcn_header
ws0.cell(r + 1, 1).border = thin
ws0.cell(r + 1, 2, "error").font = font_mcn_h
ws0.cell(r + 1, 2).fill = fill_mcn_header
ws0.cell(r + 1, 2).border = thin
for i, iid in enumerate(err_ids):
    ws0.cell(r + 2 + i, 1, iid).font = font_cell
    ws0.cell(r + 2 + i, 2, "choice supports at most 26 options in letter readout").font = font_cell
    ws0.cell(r + 2 + i, 1).border = thin
    ws0.cell(r + 2 + i, 2).border = thin

ws0.column_dimensions['A'].width = 44
ws0.column_dimensions['B'].width = 58
for col in range(3, 8):
    ws0.column_dimensions[get_column_letter(col)].width = 14

SHEET_META = [
    ("JevBench", "jevbench", "JevBench (org test)"),
    ("OpenDecisions", "opendecisions", "OpenDecisions (full)"),
    ("Fast Decisions", "fastdecisions", "Fast Decisions (test)"),
]

for sheet_name, key, title in SHEET_META:
    ws = wb.create_sheet(sheet_name)
    ws.sheet_view.showGridLines = False
    block = DATA[key]

    ws.merge_cells('A1:D1')
    ws['A1'] = f"BTP pairwise matrices — {title}"
    ws['A1'].font = font_title
    ws['A1'].fill = fill_title
    ws['A1'].alignment = Alignment(horizontal='left', vertical='center')
    ws.row_dimensions[1].height = 28

    ws.merge_cells('A2:D2')
    ws['A2'] = "Models on axes: Jev · Luna · AnyJev L0    |    Cell (Mi, Mj) = metric between row model and column model"
    ws['A2'].font = font_sub
    ws['A2'].fill = fill_subtitle
    ws.row_dimensions[2].height = 20

    ws.merge_cells('A3:D3')
    ws['A3'] = block["n_note"]
    ws['A3'].font = font_note
    ws['A3'].fill = fill_note
    ws['A3'].alignment = left
    ws.row_dimensions[3].height = 34

    row = 5
    for mkey, mtitle in METRIC_ORDER:
        row = write_matrix(ws, row, mkey, mtitle, block[mkey])

    row = write_mcnemar(ws, row, block["mcnemar"])

    ws.column_dimensions['A'].width = 38
    for col in range(2, 8):
        ws.column_dimensions[get_column_letter(col)].width = 16
    ws.freeze_panes = 'A4'

wb.save(OUT)
print("Wrote", OUT)
print("size", OUT.stat().st_size)
