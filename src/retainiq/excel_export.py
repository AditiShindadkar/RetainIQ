"""Step 7: Excel / Google Sheets MIS workbook.

Tabs
  Summary  - headline KPIs and the risk-group table
  Pivot    - churn rate & customer count by Contract x Tenure group, built with
             COUNTIFS / AVERAGEIFS formulas so it recalculates if Data changes
             (works the same in Excel and Google Sheets)
  Lookup   - type a customerID -> XLOOKUP returns risk group, score and contract
  Data     - one row per customer (the source for every formula)
"""
from __future__ import annotations

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from . import config
from .cleaning import TENURE_LABELS

HEADER_FILL = PatternFill("solid", fgColor="1F4E5F")
HEADER_FONT = Font(bold=True, color="FFFFFF")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

DATA_COLS = ["customerID", "Contract", "tenure_group", "tenure", "PaymentMethod",
             "MonthlyCharges", "Churn", "churn_score", "risk_group"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]


def _header(ws, row, values, start_col=1):
    for j, v in enumerate(values):
        c = ws.cell(row=row, column=start_col + j, value=v)
        c.fill, c.font, c.border = HEADER_FILL, HEADER_FONT, BOX
        c.alignment = Alignment(horizontal="center", wrap_text=True)


def _widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def build_workbook(scored: pd.DataFrame, segments: pd.DataFrame, path=config.EXCEL_PATH) -> None:
    wb = Workbook()
    n = len(scored)
    last = n + 1  # last data row in the Data tab

    # ------------------------------------------------------------------ Data
    data = wb.active
    data.title = "Data"
    _header(data, 1, DATA_COLS)
    for r, row in enumerate(scored[DATA_COLS].itertuples(index=False), start=2):
        for c, v in enumerate(row, start=1):
            data.cell(row=r, column=c, value=round(v, 4) if isinstance(v, float) else v)
    data.freeze_panes = "A2"
    data.auto_filter.ref = f"A1:{get_column_letter(len(DATA_COLS))}{last}"
    _widths(data, [13, 16, 14, 8, 26, 10, 7, 11, 10])

    col = {name: get_column_letter(i) for i, name in enumerate(DATA_COLS, start=1)}

    def rng(name):
        return f"Data!${col[name]}$2:${col[name]}${last}"

    # --------------------------------------------------------------- Summary
    s = wb.create_sheet("Summary", 0)
    s["A1"] = "RetainIQ: Churn MIS Summary"
    s["A1"].font = Font(bold=True, size=16, color="1F4E5F")
    s["A2"] = "All numbers are live formulas over the Data tab."
    s["A2"].font = Font(italic=True, color="666666")

    kpis = [
        ("Total customers", f"=COUNTA({rng('customerID')})", "#,##0"),
        ("Churned customers", f"=SUM({rng('Churn')})", "#,##0"),
        ("Churn rate", "=B6/B5", "0.0%"),
        ("High-risk customers", f'=COUNTIF({rng("risk_group")},"High")', "#,##0"),
        ("Monthly revenue lost to churn", f'=SUMIFS({rng("MonthlyCharges")},{rng("Churn")},1)', "$#,##0"),
    ]
    _header(s, 4, ["KPI", "Value"])
    for i, (label, formula, fmt) in enumerate(kpis, start=5):
        s.cell(row=i, column=1, value=label).border = BOX
        c = s.cell(row=i, column=2, value=formula)
        c.number_format, c.border = fmt, BOX

    _header(s, 12, ["Risk group", "Customers", "Share of base", "Actual churn rate", "Avg monthly charge"])
    for i, grp in enumerate(config.RISK_LABELS, start=13):
        s.cell(row=i, column=1, value=grp)
        s.cell(row=i, column=2, value=f'=COUNTIF({rng("risk_group")},A{i})').number_format = "#,##0"
        s.cell(row=i, column=3, value=f"=B{i}/$B$5").number_format = "0.0%"
        s.cell(row=i, column=4, value=f'=AVERAGEIFS({rng("Churn")},{rng("risk_group")},A{i})').number_format = "0.0%"
        s.cell(row=i, column=5, value=f'=AVERAGEIFS({rng("MonthlyCharges")},{rng("risk_group")},A{i})').number_format = "$0.00"
        for j in range(1, 6):
            s.cell(row=i, column=j).border = BOX
    _widths(s, [32, 14, 14, 18, 20])

    # ----------------------------------------------------------------- Pivot
    p = wb.create_sheet("Pivot", 1)
    p["A1"] = "Churn rate: Contract x Tenure group"
    p["A1"].font = Font(bold=True, size=14, color="1F4E5F")
    _header(p, 3, ["Contract \\ Tenure"] + TENURE_LABELS + ["All tenures"])
    for i, contract in enumerate(CONTRACTS, start=4):
        p.cell(row=i, column=1, value=contract).font = Font(bold=True)
        for j, tg in enumerate(TENURE_LABELS, start=2):
            h = get_column_letter(j)
            p.cell(row=i, column=j,
                   value=f'=IFERROR(AVERAGEIFS({rng("Churn")},{rng("Contract")},$A{i},{rng("tenure_group")},{h}$3),"")'
                   ).number_format = "0.0%"
        p.cell(row=i, column=6, value=f'=AVERAGEIFS({rng("Churn")},{rng("Contract")},$A{i})').number_format = "0.0%"
    p.conditional_formatting.add("B4:E6", ColorScaleRule(start_type="min", start_color="E2EFDA",
                                                         mid_type="percentile", mid_value=50, mid_color="FFEB84",
                                                         end_type="max", end_color="F8696B"))

    p["A9"] = "Customer count: Contract x Tenure group"
    p["A9"].font = Font(bold=True, size=14, color="1F4E5F")
    _header(p, 11, ["Contract \\ Tenure"] + TENURE_LABELS + ["Total"])
    for i, contract in enumerate(CONTRACTS, start=12):
        p.cell(row=i, column=1, value=contract).font = Font(bold=True)
        for j, tg in enumerate(TENURE_LABELS, start=2):
            h = get_column_letter(j)
            p.cell(row=i, column=j,
                   value=f'=COUNTIFS({rng("Contract")},$A{i},{rng("tenure_group")},{h}$11)').number_format = "#,##0"
        p.cell(row=i, column=6, value=f"=SUM(B{i}:E{i})").number_format = "#,##0"
    p["A16"] = ("Tip: for a native PivotTable, select the Data tab -> Insert -> PivotTable -> "
                "Rows: Contract, Columns: tenure_group, Values: Average of Churn.")
    p["A16"].font = Font(italic=True, color="666666")
    _widths(p, [22, 15, 15, 15, 15, 14])

    # ---------------------------------------------------------------- Lookup
    lk = wb.create_sheet("Lookup", 2)
    lk["A1"] = "Customer risk lookup (XLOOKUP)"
    lk["A1"].font = Font(bold=True, size=14, color="1F4E5F")
    lk["A3"] = "Enter customer ID:"
    lk["A3"].font = Font(bold=True)
    lk["B3"] = scored.sort_values("churn_score", ascending=False)["customerID"].iloc[0]
    lk["B3"].fill, lk["B3"].border = INPUT_FILL, BOX

    outputs = [("Risk group", "risk_group", "@"), ("Churn score", "churn_score", "0.0%"),
               ("Contract", "Contract", "@"), ("Tenure (months)", "tenure", "0"),
               ("Monthly charge", "MonthlyCharges", "$0.00"), ("Actually churned? (1=yes)", "Churn", "0")]
    for i, (label, field, fmt) in enumerate(outputs, start=5):
        lk.cell(row=i, column=1, value=label).border = BOX
        c = lk.cell(row=i, column=2,
                    value=f'=_xlfn.XLOOKUP($B$3,{rng("customerID")},{rng(field)},"Not found")')
        c.number_format, c.border = fmt, BOX

    # Batch lookup: paste a list of IDs in column D, risk group appears in E.
    _header(lk, 4, ["Batch: customer ID", "Risk group", "Churn score"], start_col=4)
    sample = scored.sample(15, random_state=config.RANDOM_STATE)["customerID"]
    for i, cid in enumerate(sample, start=5):
        lk.cell(row=i, column=4, value=cid).fill = INPUT_FILL
        lk.cell(row=i, column=5, value=f'=_xlfn.XLOOKUP(D{i},{rng("customerID")},{rng("risk_group")},"Not found")')
        lk.cell(row=i, column=6,
                value=f'=_xlfn.XLOOKUP(D{i},{rng("customerID")},{rng("churn_score")},"")').number_format = "0.0%"
    dv = DataValidation(type="textLength", operator="equal", formula1="10",
                        error="Customer IDs look like 7590-VHVEG", showErrorMessage=False)
    lk.add_data_validation(dv)
    dv.add("B3")
    _widths(lk, [26, 18, 3, 20, 14, 12])

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def python_pivot(scored: pd.DataFrame) -> pd.DataFrame:
    """Same pivot computed in pandas, used to verify the Excel formulas."""
    return (scored.pivot_table(index="Contract", columns="tenure_group", values="Churn", aggfunc="mean")
            .reindex(index=CONTRACTS, columns=TENURE_LABELS) * 100).round(1)
