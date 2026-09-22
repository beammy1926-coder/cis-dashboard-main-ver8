"""
pages_content/ai_prediction.py
--------------------------
หน้า "AI Prediction" ของ CIS Dashboard

วิธีทดสอบหน้านี้แบบเดี่ยว (ไม่ต้องรอทีมคนอื่น):
    streamlit run preview_my_page.py
    (แล้วเลือกโมดูลนี้จาก dropdown ในไฟล์ preview_my_page.py)

ข้อมูลที่ใช้ได้ใน ctx (ดูนิยามเต็มใน common.py -> class PageContext):
    ctx.selected_ticker, ctx.stock_info, ctx.stock_daily, ctx.fin_stock, ctx.sector_peers,
    ctx.scores_df, ctx.fin_df, ctx.feat_imp_df, ctx.backtest_df, ctx.risk_hist_df,
    ctx.health_yearly_df, ctx.fair_value_yearly_df,
    ctx.current_price, ctx.change_pct, ctx.change_val, ctx.change_color, ctx.change_sign, ctx.arrow_sign

ห้ามแก้ CSS ส่วนกลางหรือ helper function ใน common.py จากไฟล์นี้ — ถ้าจำเป็นต้องแก้ ให้แจ้ง Layout Lead ก่อน

=== CHANGELOG (v5 — แก้ตามรายงานตรวจสอบทีม B) ===
- [BUG-2 High แก้แล้ว] เพิ่มการเช็ค ctx.stock_info.get('is_fallback', False) บนสุดของ render() —
  ถ้าเป็น True (ข้อมูลไม่พอเทรนโมเดลจริง) แสดงกล่องคำเตือนแทนที่ KPI/Prediction/Forecast ทั้งหมด
  ไม่ให้ผู้ใช้เห็นค่า fallback คงที่เหมือนเป็นผลโมเดลจริง
- [BUG-3 High แก้แล้ว] ย้าย reliability_low มาคำนวณก่อน status_color แล้วลดระดับ status_color
  จาก GREEN เป็น AMBER เมื่อ reliability_low=True (ก่อนหน้านี้ status_color อิง prob_up อย่างเดียว
  ทำให้ THCOM/JMART ขึ้น STRONG BUY สีเขียวทั้งที่ accuracy ต่ำกว่า baseline มาก) เพิ่ม signal_display
  (มี ⚠ ต่อท้ายเมื่อ reliability_low) ใช้เฉพาะกล่อง KPI RECOMMENDATION จุดที่สายตาเห็นก่อนสุด
- [ISSUE-4 Medium แก้แล้ว] ขยายเงื่อนไข reliability_low ให้ครอบคลุมกรณีโมเดล degenerate
  (precision=0 หรือ recall=0 เช่นเคส HANA ที่ accuracy เท่ากับ baseline พอดีแต่ไม่เคยทำนาย "ขึ้น" เลย)
  เปลี่ยนจาก accuracy < baseline (strict) เป็น accuracy <= baseline หรือ precision/recall == 0
- (คงไว้จาก v4) ขยายฟอนต์ทั่วหน้า, จัด Prediction section เป็น 2 กล่อง, เปลี่ยน emoji 🔮->📈, ลบ 🧠
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from common import fmt_mb, fmt_ratio, safe, show_chart, render_nav_footer, COMPANY_NAMES, SECTOR_MAP

GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"
BLUE = "#38BDF8"


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _section_title(text):
    return f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 16px 0 16px;">
<div><span style="font-size:15.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">{text}</span></div></div>"""


def render(ctx):
    st.markdown(f"""<div style="margin-bottom:14px;">
<div style="font-size:14.5px; color:#64748B; margin-bottom:2px;">Home / Module 4 / AI Prediction</div>
<h2 style="margin:0; font-size:23px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">AI PREDICTION</h2>
</div>""", unsafe_allow_html=True)

    # ============================================================
    # BUG-2 fix: ถ้าเป็นค่า fallback (ข้อมูลไม่พอเทรนโมเดลจริง) ห้ามแสดง KPI/กราฟเหมือนเป็นผลจริง
    # ============================================================
    if ctx.stock_info.get('is_fallback', False):
        st.markdown(f"""<div style="background:rgba(239,68,68,0.1); border:1px solid {RED}; border-radius:12px; padding:28px; text-align:center;">
<div style="font-size:21px; font-weight:bold; color:{RED}; margin-bottom:10px;">⚠ ข้อมูลไม่เพียงพอสำหรับ {ctx.selected_ticker}</div>
<p style="font-size:15px; color:#CBD5E1; line-height:1.65; margin:0; max-width:640px; margin:0 auto;">
หุ้นตัวนี้มีข้อมูลราคาย้อนหลังไม่พอสำหรับเทรนโมเดล Random Forest จริง (ต้องการอย่างน้อย Train 50 แถว และ Test 20 แถว)
ตัวเลขที่เคยแสดงในหน้านี้เป็นเพียง<b>ค่าตั้งต้นสำรอง (placeholder)</b> ไม่ใช่ผลจากการเทรนโมเดลจริงแต่อย่างใด
จึง<b style="color:{RED};">ไม่ควรใช้ประกอบการตัดสินใจลงทุน</b></p>
</div>""", unsafe_allow_html=True)
        render_nav_footer("m4", prev_page=" ⏱️ Entry Timing", next_page=" 🛡️ Risk Analysis")
        return

    # ============================================================
    # แหล่งความจริงเดียวของสถานะทำนาย — ใช้ทุกจุดในหน้า ไม่คำนวณซ้ำแยกชุด
    # ============================================================
    prob_up = safe(ctx.stock_info.get('prob_up'), 50)
    down_prob = round(100 - prob_up, 1)
    ai_score = int(round(safe(ctx.stock_info.get('ai_score'), 50)))
    acc_val = safe(ctx.stock_info.get('accuracy'), 50)
    baseline_val = safe(ctx.stock_info.get('baseline_accuracy'), acc_val)
    prec_val = safe(ctx.stock_info.get('precision'), 1)
    rec_val = safe(ctx.stock_info.get('recall'), 1)
    signal = ctx.stock_info.get('ai_signal', '-')

    # BUG-3 + ISSUE-4 fix: คำนวณ reliability_low ก่อน แล้วให้มีผลกับ status_color ทันที
    # ครอบคลุมทั้งกรณี accuracy <= baseline (เผื่อเท่ากันพอดีแบบ HANA) และกรณีโมเดล degenerate (precision/recall=0)
    reliability_low = (acc_val <= baseline_val) or (prec_val == 0) or (rec_val == 0)

    if prob_up >= 70:
        status_color = GREEN
    elif prob_up >= 50:
        status_color = AMBER
    else:
        status_color = RED

    if reliability_low and status_color == GREEN:
        status_color = AMBER  # ไม่ให้แสดงความมั่นใจเต็มที่ ถ้าโมเดลแย่กว่า/เท่ากับเกณฑ์ทายมั่ว (baseline)

    direction_th = "ขาขึ้น" if prob_up >= 50 else "ขาลง"
    if acc_val > baseline_val:
        baseline_note = "สูงกว่า"
    elif acc_val == baseline_val:
        baseline_note = "เท่ากับ"
    else:
        baseline_note = "ต่ำกว่า"

    # signal_display มี ⚠ ต่อท้ายเมื่อ reliability_low — ใช้เฉพาะกล่อง KPI RECOMMENDATION (จุดที่เห็นก่อนสุด)
    # ส่วน signal เดิมใช้กับประโยคอธิบายในกล่อง Prediction ตามที่รายงานแนะนำ
    signal_display = f"{signal} ⚠" if reliability_low else signal

    st.markdown("<div style='margin-top:0px;'></div>", unsafe_allow_html=True)

    # ============================================================
    # 1) OVERVIEW — แถบ KPI ใหญ่ ตอบคำถาม "สรุปแล้วตัวเลขคืออะไร"
    # ============================================================
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px 12px; text-align:center; min-height:122px;">
<div style="font-size:12.5px; font-weight:bold; color:#64748B; letter-spacing:1px;">PRICE</div>
<div style="font-size:34px; font-weight:bold; color:#F8FAFC; line-height:1.15; margin-top:4px;">{ctx.current_price:,.2f}</div>
<div style="font-size:13px; font-weight:bold; color:{ctx.change_color};">{ctx.change_val:+.2f} ({ctx.change_pct:+.2f}%) {ctx.arrow_sign}</div>
</div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px 12px; text-align:center; min-height:122px;">
<div style="font-size:12.5px; font-weight:bold; color:#64748B; letter-spacing:1px;">DIRECTION (10D)</div>
<div style="font-size:34px; font-weight:bold; color:{status_color}; line-height:1.15; margin-top:4px;">{direction_th}</div>
<div style="font-size:13px; color:#64748B;">10 Trading Days</div>
</div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px 12px; text-align:center; min-height:122px;">
<div style="font-size:12.5px; font-weight:bold; color:#64748B; letter-spacing:1px;">PROBABILITY</div>
<div style="font-size:34px; font-weight:bold; color:{status_color}; line-height:1.15; margin-top:4px;">{prob_up:.0f}%</div>
<div style="font-size:13px; color:#64748B;">Down: {down_prob:.0f}%</div>
</div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px 12px; text-align:center; min-height:122px;">
<div style="font-size:12.5px; font-weight:bold; color:#64748B; letter-spacing:1px;">SCORE</div>
<div style="font-size:34px; font-weight:bold; color:{status_color}; line-height:1.15; margin-top:4px;">{ai_score}<span style="font-size:17px; color:#64748B;">/100</span></div>
<div style="font-size:13px; color:#64748B;">Prediction Score</div>
</div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid {status_color}; border-radius:12px; padding:16px 12px; text-align:center; min-height:122px;">
<div style="font-size:12.5px; font-weight:bold; color:#64748B; letter-spacing:1px;">RECOMMENDATION</div>
<div style="font-size:22px; font-weight:bold; color:{status_color}; line-height:1.3; margin-top:6px;">{signal_display}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)

    # ============================================================
    # 2) PREDICTION — ตอบคำถาม "ทำไมถึงเป็นแบบนี้" (ภาพรวมความน่าจะเป็น)
    #    ซ้าย = gauge (ใหญ่, ไม่มีข้อความซ้อนในวงกลม) | ขวา = คำอธิบาย (กล่องสว่างกว่า, ฟอนต์ใหญ่)
    # ============================================================
    st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:10px 16px; margin-bottom:10px;">
<span style="font-size:15.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">📈 PREDICTION</span>
</div>""", unsafe_allow_html=True)

    _t = min(1, max(0, prob_up / 100))
    _gx = 50 - 40 * np.cos(np.pi * _t)
    _gy = 50 - 40 * np.sin(np.pi * _t)

    warn_line = ""
    if reliability_low:
        warn_line = f"""<div style="font-size:13px; color:{RED}; background:rgba(239,68,68,0.1); border:1px solid {RED}; border-radius:8px; padding:9px 12px; margin-top:12px;">
⚠ ความแม่นยำของโมเดลต่ำกว่าหรือเท่ากับเกณฑ์เปรียบเทียบ (baseline) สำหรับหุ้นตัวนี้ — ควรใช้ผลทำนายนี้ด้วยความระมัดระวังเป็นพิเศษ</div>"""

    pred_left, pred_right = st.columns([1, 1.6])

    with pred_left:
        st
