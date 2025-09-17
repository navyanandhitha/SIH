# app.py
# Streamlit: AI-based Drop-out Prediction & Counseling (Rule-first, ML-assist optional)
# Run: streamlit run app.py

import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

# ---------- Page config ----------
st.set_page_config(
    page_title="Early Warning: Student Drop-out Risk",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .small {font-size:0.9rem; color:#666}
    .pill {padding:3px 10px; border-radius:999px; font-weight:600}
    .pill.red {background:#ffe5e5; color:#b00020}
    .pill.amber {background:#fff3e0; color:#8a4b00}
    .pill.green {background:#e8f5e9; color:#1b5e20}
    .metric-card {border:1px solid #eee; border-radius:14px; padding:14px; background:#fff}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- CSV Readers ----------
@st.cache_data
def _read_csv(upload, **kwargs):
    return pd.read_csv(upload, **kwargs)

def read_csv(upload, **kwargs):
    if upload is None:
        return None
    try:
        return _read_csv(upload, **kwargs)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

# ---------- Small helpers ----------
def risk_emoji(level: str) -> str:
    return {"RED":"🟥","AMBER":"🟧","GREEN":"🟩"}.get(level, "🟩")

def wa_link(phone: str, text: str):
    if not isinstance(phone, str): phone = ""
    digits = "".join(ch for ch in phone if ch.isdigit() or ch == "+")
    if digits.startswith("+"): digits = digits[1:]
    import urllib.parse as up
    return f"https://wa.me/{digits}?text={up.quote(text)}" if digits else ""

def is_email(s: str) -> bool:
    if not isinstance(s, str): return False
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s.strip()) is not None

# ---------- Sidebar: Inputs & Config ----------
st.sidebar.title("⚙ Configure")
st.sidebar.caption("Upload CSVs ➜ set thresholds ➜ view dashboard ➜ send emails")

with st.sidebar.expander("1) Upload CSV Files", expanded=True):
    students_csv    = st.file_uploader("students.csv (required)", type=["csv"], key="students")
    attendance_csv  = st.file_uploader("attendance.csv (required)", type=["csv"], key="attendance")
    assess_csv      = st.file_uploader("assessments.csv (required)", type=["csv"], key="assess")
    fees_csv        = st.file_uploader("fees.csv (optional)", type=["csv"], key="fees")

with st.sidebar.expander("2) Rule Thresholds (editable)", expanded=False):
    st.markdown("Attendance")
    min_attn_green = st.slider("Min attendance for GREEN (%)", 60, 100, 85, step=1)
    min_attn_amber = st.slider("Min attendance for AMBER (%)", 40, min_attn_green, 70, step=1)

    st.markdown("---\n*Assessment Scores*")
    min_avg_green  = st.slider("Min avg score for GREEN (%)", 40, 100, 65, step=1)
    min_avg_amber  = st.slider("Min avg score for AMBER (%)", 20, min_avg_green, 50, step=1)
    min_trend_drop = st.slider("Significant downward trend (points)", 1, 30, 10, step=1)

    st.markdown("---\n*Backlogs / Attempts*")
    max_backlogs_green = st.slider("Max backlogs for GREEN", 0, 10, 0, step=1)
    max_backlogs_amber = st.slider("Max backlogs for AMBER", max_backlogs_green, 15, 2, step=1)
    attempts_limit     = st.slider("Attempts exhausted threshold (>=)", 1, 10, 3, step=1)

    st.markdown("---\n*Fees*")
    fee_delay_amber = st.slider("Fee overdue days for AMBER", 0, 120, 15, step=1)
    fee_delay_red   = st.slider("Fee overdue days for RED", fee_delay_amber, 240, 45, step=1)

with st.sidebar.expander("3) Notifications (who & cadence)"):
    notify_who = st.multiselect("Send emails to", ["Mentors", "Students"], default=["Mentors","Students"])
    include_levels = st.multiselect("Email which risk levels", ["RED","AMBER","GREEN"], default=["RED","AMBER"])
    schedule_hint = st.selectbox("Intended cadence (note in email footer)", ["Weekly (Mon 9AM)", "Fortnightly (Mon 9AM)", "Monthly (1st, 9AM)"], index=0)

with st.sidebar.expander("4) SMTP Settings (email sending) — required"):
    smtp_host = st.text_input("SMTP Host", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, value=587, step=1)
    use_tls   = st.checkbox("Use STARTTLS (recommended)", value=True)
    sender_name  = st.text_input("Sender Name", value="Early Warning Dashboard")
    sender_email = st.text_input("Sender Email (login user)", value="")
    smtp_user    = st.text_input("SMTP Username", value="", help="Often same as Sender Email")
    smtp_pass    = st.text_input("SMTP Password / App Password", type="password", value="")
    dry_run      = st.checkbox("Dry-run (preview only, do not send)", value=True)

with st.sidebar.expander("📌 Notes"):
    st.caption("• Use an App Password for Gmail/Outlook.\n• Keep this app private since it handles email credentials.\n• Dry-run shows exactly who would be emailed without sending.")

# ---------- Data Loading ----------
def _require_cols(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"'{name}' is missing columns: {missing}")
        return False
    return True

def load_all():
    students = read_csv(students_csv)
    attendance = read_csv(attendance_csv)
    assessments = read_csv(assess_csv)
    fees = read_csv(fees_csv) if fees_csv else None
    if students is None or attendance is None or assessments is None:
        st.warning("Please upload the required CSVs.")
        return None, None, None, None

    if not _require_cols(students, ["student_id","name","program","semester","mentor_name","mentor_email","student_email","guardian_phone"], "students.csv"):
        return None, None, None, None
    if not _require_cols(attendance, ["student_id","date","present"], "attendance.csv"):
        return None, None, None, None
    if not _require_cols(assessments, ["student_id","subject_code","test_date","score","max_score","attempt_number"], "assessments.csv"):
        return None, None, None, None
    if fees is not None and not _require_cols(fees, ["student_id","fee_due_date","amount_due","amount_paid","last_payment_date"], "fees.csv"):
        return None, None, None, None

    attendance["date"] = pd.to_datetime(attendance["date"], errors="coerce")
    assessments["test_date"] = pd.to_datetime(assessments["test_date"], errors="coerce")
    if fees is not None:
        fees["fee_due_date"] = pd.to_datetime(fees["fee_due_date"], errors="coerce")
        fees["last_payment_date"] = pd.to_datetime(fees["last_payment_date"], errors="coerce")
    return students, attendance, assessments, fees

students, attendance, assessments, fees = load_all()
if students is None:
    st.stop()

# ---------- Feature Engineering ----------
def aggregate_attendance(att_df):
    att = att_df.groupby("student_id", as_index=False).agg(
        total_days=("present","count"),
        present_days=("present","sum")
    )
    att["attendance_pct"] = (att["present_days"] / att["total_days"] * 100).round(1)
    return att

def scores_features(assess_df):
    df = assess_df.copy()
    df["score_pct"] = (df["score"] / df["max_score"] * 100).clip(0, 100)

    avg = df.groupby("student_id", as_index=False).agg(
        avg_score_pct=("score_pct","mean"),
        tests_taken=("score_pct","count"),
        max_attempt=("attempt_number","max")
    )

    def trend_calc(g):
        g = g.sort_values("test_date")
        last3 = g["score_pct"].tail(3).mean() if len(g)>=1 else np.nan
        prev3 = g["score_pct"].iloc[-6:-3].mean() if len(g)>=6 else np.nan
        trend = last3 - prev3 if not (np.isnan(last3) or np.isnan(prev3)) else np.nan
        return pd.Series({"trend_last3_vs_prev3": trend})

    trend = df.groupby("student_id").apply(trend_calc).reset_index()
    subj_avg = df.groupby(["student_id","subject_code"], as_index=False)["score_pct"].mean()
    backlogs = subj_avg[subj_avg["score_pct"] < 35].groupby("student_id", as_index=False).size()
    backlogs = backlogs.rename(columns={"size":"backlogs"}).fillna({"backlogs":0})
    return avg.merge(trend, on="student_id", how="left").merge(backlogs, on="student_id", how="left").fillna({"backlogs":0})

def fees_features(fees_df):
    if fees_df is None or fees_df.empty:
        return pd.DataFrame(columns=["student_id","fee_overdue_days","fee_due_balance"])

    today = pd.Timestamp.today().normalize()
    f = fees_df.copy()
    f["fee_due_balance"] = (f["amount_due"] - f["amount_paid"]).clip(lower=0)
    f["fee_overdue_days"] = np.where(
        (f["fee_due_balance"] > 0) & (f["fee_due_date"].notna()),
        (today - f["fee_due_date"]).dt.days.clip(lower=0),
        0
    )
    agg = f.groupby("student_id", as_index=False).agg(
        fee_overdue_days=("fee_overdue_days","max"),
        fee_due_balance=("fee_due_balance","sum")
    )
    return agg

att = aggregate_attendance(attendance)
sc  = scores_features(assessments)
ff  = fees_features(fees)

# ---------- Merge ----------
df = (
    students.merge(att, on="student_id", how="left")
            .merge(sc, on="student_id", how="left")
            .merge(ff, on="student_id", how="left")
)

for col in ["attendance_pct","avg_score_pct","tests_taken","max_attempt","trend_last3_vs_prev3","backlogs","fee_overdue_days","fee_due_balance"]:
    if col not in df.columns:
        df[col] = np.nan

df[["backlogs","tests_taken","max_attempt","fee_overdue_days"]] = df[["backlogs","tests_taken","max_attempt","fee_overdue_days"]].fillna(0).astype(int)
df["avg_score_pct"] = df["avg_score_pct"].fillna(0).round(1)
df["attendance_pct"] = df["attendance_pct"].fillna(0).round(1)

# ---------- Rule Engine ----------
def apply_rules(row):
    points = 0
    notes = []

    # Attendance
    if row.attendance_pct < min_attn_amber:
        points += 3; notes.append("Low attendance")
    elif row.attendance_pct < min_attn_green:
        points += 1; notes.append("Borderline attendance")

    # Scores
    if row.avg_score_pct < min_avg_amber:
        points += 3; notes.append("Low avg score")
    elif row.avg_score_pct < min_avg_green:
        points += 1; notes.append("Borderline avg score")

    # Trend
    if not pd.isna(row.trend_last3_vs_prev3) and row.trend_last3_vs_prev3 <= -min_trend_drop:
        points += 2; notes.append("Score trending down")

    # Backlogs
    if row.backlogs >= max_backlogs_amber:
        points += 2; notes.append("Multiple backlogs")
    elif row.backlogs > max_backlogs_green:
        points += 1; notes.append("Some backlogs")

    # Attempts
    if row.max_attempt >= attempts_limit:
        points += 2; notes.append(f"Attempts ≥ {attempts_limit}")

    # Fees
    if row.fee_overdue_days >= fee_delay_red:
        points += 2; notes.append("Severe fee delay")
    elif row.fee_overdue_days >= fee_delay_amber:
        points += 1; notes.append("Fee delay")

    if points >= 6: level = "RED"
    elif points >= 3: level = "AMBER"
    else: level = "GREEN"

    return pd.Series({"risk_points": points, "risk_level": level, "risk_reasons": "; ".join(notes)})

risk_cols = df.apply(apply_rules, axis=1)
df = pd.concat([df, risk_cols], axis=1)

# ---------- UI ----------
st.title("📉 Early Warning System: Drop-out Risk")
st.caption("Rule-first, transparent logic with optional ML-assisted ranking. Built for low-cost early intervention.")

# Filters
colF1, colF2, colF3, colF4 = st.columns(4)
with colF1:
    prog = st.multiselect("Program", sorted(df["program"].dropna().unique().tolist()), placeholder="All")
with colF2:
    sem = st.multiselect("Semester", sorted(df["semester"].dropna().unique().tolist()), placeholder="All")
with colF3:
    mentor = st.multiselect("Mentor", sorted(df["mentor_name"].dropna().unique().tolist()), placeholder="All")
with colF4:
    riskpick = st.multiselect("Risk Level", ["RED","AMBER","GREEN"], default=["RED","AMBER"], placeholder="Any")

mask = pd.Series(True, index=df.index)
if prog:   mask &= df["program"].isin(prog)
if sem:    mask &= df["semester"].isin(sem)
if mentor: mask &= df["mentor_name"].isin(mentor)
if riskpick: mask &= df["risk_level"].isin(riskpick)
view = df[mask].copy()

# Metrics
c1, c2, c3, c4 = st.columns(4)
for level, col in zip(["RED","AMBER","GREEN"], [c1,c2,c3]):
    cnt = int((view["risk_level"]==level).sum())
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="pill {level.lower()}">{risk_emoji(level)} {level}</div>'
            f'<h2 style="margin:6px 0">{cnt}</h2><div class="small">students</div></div>',
            unsafe_allow_html=True
        )
with c4:
    total = len(view)
    st.markdown(
        f'<div class="metric-card"><div class="pill green">👥 TOTAL</div>'
        f'<h2 style="margin:6px 0">{total}</h2><div class="small">in filtered view</div></div>',
        unsafe_allow_html=True
    )

# Charts
left, right = st.columns((2,3))
with left:
    risk_count = view.groupby("risk_level").size().reset_index(name="count")
    if not risk_count.empty:
        chart = alt.Chart(risk_count).mark_bar().encode(
            x=alt.X("risk_level:N", sort=["RED","AMBER","GREEN"]),
            y="count:Q",
            color=alt.Color("risk_level:N", scale=alt.Scale(domain=["RED","AMBER","GREEN"], range=["#e53935","#fb8c00","#43a047"])),
            tooltip=["risk_level","count"]
        ).properties(height=240, title="Risk distribution")
        st.altair_chart(chart, use_container_width=True)

with right:
    scatter = view.copy()
    sc_chart = alt.Chart(scatter).mark_circle(size=80).encode(
        x=alt.X("attendance_pct:Q", title="Attendance (%)"),
        y=alt.Y("avg_score_pct:Q", title="Average Score (%)"),
        color=alt.Color("risk_level:N", scale=alt.Scale(domain=["RED","AMBER","GREEN"], range=["#e53935","#fb8c00","#43a047"])),
        tooltip=["student_id","name","program","semester","attendance_pct","avg_score_pct","backlogs","risk_level"]
    ).properties(height=240, title="Attendance vs Score (colored by risk)")
    st.altair_chart(sc_chart, use_container_width=True)

# Table
def render_table(df_in):
    show = df_in.copy()
    # Sort: RED -> AMBER -> GREEN, then higher points
    if "risk_level" in show.columns:
        cat = pd.api.types.CategoricalDtype(categories=["RED","AMBER","GREEN"], ordered=True)
        show["risk_level"] = show["risk_level"].astype(cat)
    sort_cols = [c for c in ["risk_level","risk_points"] if c in show.columns]
    if sort_cols:
        asc = [True if c == "risk_level" else False for c in sort_cols]
        show = show.sort_values(sort_cols, ascending=asc, kind="mergesort")
    # Display cols
    if "risk_level" in show.columns:
        show.insert(0, "Risk", show["risk_level"].astype(str).map(lambda x: f"{risk_emoji(x)} {x}"))
    else:
        show.insert(0, "Risk", "🟩 GREEN")
    cols = ["Risk","student_id","name","program","semester","mentor_name","attendance_pct","avg_score_pct",
            "trend_last3_vs_prev3","backlogs","max_attempt","fee_overdue_days","risk_points","risk_reasons"]
    for c in cols:
        if c not in show.columns:
            show[c] = np.nan
    st.dataframe(show[cols], use_container_width=True, hide_index=True)

st.subheader("Student Risk Table")
render_table(view)

# ---------- Email Templates ----------
def email_subject_mentor(r):
    return f"[Early Warning] {r['name']} ({r['student_id']}) - {r['risk_level']} Risk"

def email_body_mentor(r):
    return "\n".join([
        f"Dear {r['mentor_name']},",
        "",
        f"Student: {r['name']} ({r['student_id']})",
        f"Program/Sem: {r['program']} / {r['semester']}",
        f"Risk Level: {r['risk_level']} ({r['risk_points']} pts)",
        f"Reasons: {r['risk_reasons']}",
        "",
        f"Key metrics:",
        f"• Attendance: {r['attendance_pct']}%",
        f"• Avg Score: {r['avg_score_pct']}%",
        f"• Backlogs: {r['backlogs']}",
        f"• Max Attempts: {r['max_attempt']}",
        f"• Fee overdue: {r['fee_overdue_days']} days",
        "",
        "Suggested next steps:",
        "- Mentor connect within 48 hours",
        "- Short remedial plan (attendance + topic gaps)",
        "- Guardian call if RED",
        "",
        f"Cadence: {schedule_hint} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "— Automated Early Warning Dashboard",
    ])

def email_subject_student(r):
    return f"[Academic Support] Your Progress Update - {r['risk_level']} Risk"

def email_body_student(r):
    tips = []
    if r["attendance_pct"] < min_attn_green: tips.append("Attend all classes for the next two weeks and meet course faculty.")
    if r["avg_score_pct"] < min_avg_green: tips.append("Schedule 30–45 mins daily for revision; attempt past papers weekly.")
    if r["backlogs"] > 0: tips.append("Create a backlog clearance plan with mentor; track weekly topics.")
    if r["fee_overdue_days"] >= fee_delay_amber: tips.append("Contact accounts to arrange a payment plan.")
    if not tips:
        tips.append("Keep up the good work and stay consistent!")
    tip_lines = "\n".join([f"- {t}" for t in tips])

    return "\n".join([
        f"Hi {r['name']},",
        "",
        f"This is a friendly update about your current academic status.",
        f"Risk Level: {r['risk_level']} ({r['risk_points']} pts)",
        f"Reasons noted: {r['risk_reasons'] or '—'}",
        "",
        "Key metrics:",
        f"• Attendance: {r['attendance_pct']}%",
        f"• Avg Score: {r['avg_score_pct']}%",
        f"• Backlogs: {r['backlogs']}",
        f"• Attempts (max): {r['max_attempt']}",
        "",
        "Suggested next steps tailored for you:",
        tip_lines,
        "",
        f"Cadence: {schedule_hint} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "If you need any help, reply to this email or reach out to your mentor.",
        "",
        "— Academic Support",
    ])

# ---------- Alerts view (for export & email) ----------
alerts_view = view.copy()
alerts_view["email_subject_mentor"]  = alerts_view.apply(email_subject_mentor, axis=1)
alerts_view["email_body_mentor"]     = alerts_view.apply(email_body_mentor, axis=1)
alerts_view["email_subject_student"] = alerts_view.apply(email_subject_student, axis=1)
alerts_view["email_body_student"]    = alerts_view.apply(email_body_student, axis=1)
alerts_view["guardian_wa_link"]      = alerts_view.apply(
    lambda r: wa_link(
        r.get("guardian_phone",""),
        f"Dear Parent/Guardian of {r['name']} ({r['student_id']}), please connect regarding academic progress. Risk: {r['risk_level']}. Mentor: {r['mentor_name']}."
    ),
    axis=1
)

# Sort alerts so RED first, then by points desc
alerts_sorted = alerts_view.copy()
alerts_sorted["_risk_order"] = pd.Categorical(alerts_sorted["risk_level"], categories=["RED","AMBER","GREEN"], ordered=True)
alerts_sorted = alerts_sorted.sort_values(["_risk_order","risk_points"], ascending=[True, False])

st.subheader("📬 Alerts (preview & export)")
export_cols = [
    "risk_level","risk_points","student_id","name","program","semester","mentor_name","mentor_email",
    "student_email","guardian_phone","attendance_pct","avg_score_pct","backlogs","max_attempt",
    "fee_overdue_days","risk_reasons","email_subject_mentor","email_body_mentor",
    "email_subject_student","email_body_student","guardian_wa_link"
]
st.dataframe(alerts_sorted[export_cols], use_container_width=True, hide_index=True)
st.download_button(
    "⬇ Download Alerts CSV",
    data=alerts_sorted[export_cols].to_csv(index=False).encode("utf-8"),
    file_name="alerts_export.csv", mime="text/csv"
)

# ---------- Email Sender ----------
def send_email_batch(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Sends emails to mentors/students depending on 'notify_who'.
    Respects 'include_levels' and 'dry_run'.
    Returns a dataframe with status columns.
    """
    out = df_rows.copy()
    out["send_to_mentor_status"]  = ""
    out["send_to_student_status"] = ""

    # Validate SMTP inputs (only if not dry-run)
    if not dry_run:
        if not (smtp_host and smtp_port and sender_email and smtp_user and smtp_pass):
            st.error("SMTP details are incomplete. Please fill all fields or enable Dry-run.")
            return out

    # Build a connection if needed
    server = None
    try:
        if not dry_run:
            server = smtplib.SMTP(smtp_host, int(smtp_port), timeout=30)
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_pass)

        progress = st.progress(0)
        total = len(out)
        for i, (_, r) in enumerate(out.iterrows(), start=1):
            # Mentor
            if "Mentors" in notify_who and r["risk_level"] in include_levels:
                to_addr = (r.get("mentor_email") or "").strip()
                if is_email(to_addr):
                    sub = r["email_subject_mentor"]; body = r["email_body_mentor"]
                    status = "DRY-RUN (not sent)"
                    if not dry_run:
                        try:
                            msg = MIMEText(body, "plain", "utf-8")
                            msg["Subject"] = sub
                            msg["From"] = formataddr((sender_name, sender_email))
                            msg["To"] = to_addr
                            server.sendmail(sender_email, [to_addr], msg.as_string())
                            status = "SENT"
                        except Exception as e:
                            status = f"ERROR: {e}"
                    out.at[_, "send_to_mentor_status"] = status
                else:
                    out.at[_, "send_to_mentor_status"] = "SKIPPED: invalid/missing email"

            # Student
            if "Students" in notify_who and r["risk_level"] in include_levels:
                to_addr = (r.get("student_email") or "").strip()
                if is_email(to_addr):
                    sub = r["email_subject_student"]; body = r["email_body_student"]
                    status = "DRY-RUN (not sent)"
                    if not dry_run:
                        try:
                            msg = MIMEText(body, "plain", "utf-8")
                            msg["Subject"] = sub
                            msg["From"] = formataddr((sender_name, sender_email))
                            msg["To"] = to_addr
                            server.sendmail(sender_email, [to_addr], msg.as_string())
                            status = "SENT"
                        except Exception as e:
                            status = f"ERROR: {e}"
                    out.at[_, "send_to_student_status"] = status
                else:
                    out.at[_, "send_to_student_status"] = "SKIPPED: invalid/missing email"

            progress.progress(min(i / total, 1.0))
        if dry_run:
            st.info("Dry-run complete. No emails were sent.")
        else:
            st.success("Email batch complete.")
    finally:
        try:
            if server: server.quit()
        except Exception:
            pass
    return out

st.markdown("---")
st.subheader("📧 Send Emails Now")
st.caption("Choose risk levels & recipients in the sidebar. Enter SMTP details. Use Dry-run first.")

col_send_a, col_send_b = st.columns([1,2])
with col_send_a:
    send_btn = st.button("Send Emails Now", type="primary", use_container_width=True)
with col_send_b:
    st.write("")

if send_btn:
    # Only selected levels included
    to_send = alerts_sorted[alerts_sorted["risk_level"].isin(include_levels)].copy()
    if to_send.empty:
        st.warning("No rows match the selected risk levels.")
    else:
        results = send_email_batch(to_send)
        st.write("Batch results (latest run):")
        st.dataframe(
            results[[
                "risk_level","risk_points","student_id","name",
                "mentor_email","send_to_mentor_status",
                "student_email","send_to_student_status"
            ]],
            use_container_width=True, hide_index=True
        )

# ---------- Optional ML Assist ----------
with st.expander("🤖 ML-assisted Ranking (Logistic Regression)", expanded=False):
    ml_assist = st.checkbox("Enable ML-assisted ranking (learns from labels)", value=False)
    train_size = st.slider("Train size (%)", 50, 90, 70, step=5)
    if ml_assist:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        tmp = df.copy()
        tmp["label"] = tmp["risk_level"].map({"RED":1,"AMBER":1,"GREEN":0})
        feats = ["attendance_pct","avg_score_pct","trend_last3_vs_prev3","backlogs","max_attempt","fee_overdue_days"]
        X = tmp[feats].fillna(0.0).values
        y = tmp["label"].values

        if len(np.unique(y)) < 2 or (y == 1).sum() < 2 or (y == 0).sum() < 2:
            st.info("Not enough variation in labels for ML training (need both at-risk and safe). Adjust thresholds or add data.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size/100.0, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_s, y_train)

            prob = clf.predict_proba(scaler.transform(tmp[feats].fillna(0.0).values))[:,1]
            tmp["ml_risk_prob"] = (prob*100).round(1)

            meld = view.merge(tmp[["student_id","ml_risk_prob"]], on="student_id", how="left") \
                       .sort_values("ml_risk_prob", ascending=False)
            st.write("Top students by ML risk probability (within current filters):")
            st.dataframe(
                meld[["student_id","name","program","semester","risk_level","risk_points","ml_risk_prob"]].head(25),
                use_container_width=True, hide_index=True
            )

            coef = pd.Series(clf.coef_[0], index=feats).sort_values(key=lambda s: s.abs(), ascending=False).round(3)
            st.write("Model coefficients (magnitude = importance, sign = direction):")
            st.dataframe(coef.rename("coef"))

