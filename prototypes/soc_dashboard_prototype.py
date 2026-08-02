"""
PROTOTYPE — throwaway Streamlit dashboard to explore what's visually possible
for a cybersecurity SOC monitor with BETH-style event data.

Three variants, switchable via sidebar.
Run: .venv\Scripts\Activate.ps1; streamlit run prototypes\soc_dashboard_prototype.py

DO NOT SHIP — fold the winning variant into the real project and delete this file.
"""

import random
import time
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOC Monitor — Prototype",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Simulated BETH-style data ────────────────────────────────────────────────
PROCESS_NAMES_BENIGN = [
    "systemd", "sshd", "bash", "systemd-logind", "dbus-daemon",
    "cron", "rsyslogd", "accounts-daemon", "polkitd", "networkd-dispat",
    "snapd", "containerd", "dockerd", "kubelet", "journald",
]

PROCESS_NAMES_SUSPICIOUS = [
    "wget", "curl", "nc", "python3", "chmod", "scp",
    "whoami", "id", "uname", "cat /etc/passwd", "nmap",
]

PROCESS_NAMES_MALICIOUS = [
    "xmrig", "minerd", "crypto-miner", "botnet-client", "reverse-shell",
    "keylogger", "rootkit-install", "data-exfil", "ransomware-encrypt",
]

MITRE_TACTICS = [
    "Reconnaissance", "Resource Development", "Initial Access",
    "Execution", "Persistence", "Privilege Escalation",
    "Defense Evasion", "Credential Access", "Discovery",
    "Lateral Movement", "Collection", "Command and Control",
    "Exfiltration", "Impact",
]

MITRE_TECHNIQUES = {
    "Execution": ["T1059", "T1203", "T1106"],
    "Persistence": ["T1547", "T1053", "T1543"],
    "Defense Evasion": ["T1070", "T1564", "T1027"],
    "Command and Control": ["T1071", "T1105", "T1571"],
    "Exfiltration": ["T1041", "T1567", "T1020"],
    "Impact": ["T1486", "T1496", "T1529"],
}


@st.cache_data
def generate_event_data(n_events=2000, seed=42):
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 7, 15, 2, 14, 0)
    data = []

    # Generate benign background
    for i in range(n_events):
        ts = base_time + timedelta(seconds=int(i * 0.8 + rng.exponential(0.3)))
        proc = rng.choice(PROCESS_NAMES_BENIGN)
        user_id = rng.choice([0, 0, 0, 1, 2, 3])
        event_id = rng.integers(1, 50)
        args_num = rng.integers(0, 8)
        return_val = rng.choice([0, 0, 0, 0, -1, 1])
        sus = 0
        evil = 0

        # Inject suspicious events
        if i > 400 and i % rng.integers(40, 80) == 0:
            proc = rng.choice(PROCESS_NAMES_SUSPICIOUS)
            user_id = rng.integers(1000, 1005)
            sus = 1

        # Inject attack burst (botnet node install)
        if 800 < i < 1000:
            if i % rng.integers(3, 8) == 0:
                proc = rng.choice(PROCESS_NAMES_BENIGN) if rng.random() < 0.4 else rng.choice(PROCESS_NAMES_MALICIOUS)
                user_id = 1001
                sus = 1
                evil = 1
                args_num = rng.integers(3, 15)

        data.append({
            "timestamp": ts,
            "processName": proc,
            "userId": user_id,
            "eventId": int(event_id),
            "argsNum": int(args_num),
            "returnValue": int(return_val),
            "sus": sus,
            "evil": evil,
            "mountNamespace": 4026531840 if user_id >= 1000 else rng.choice([4026531840, 4026531836]),
        })

    df = pd.DataFrame(data)
    df["timestamp_numeric"] = (df["timestamp"] - base_time).dt.total_seconds()
    df["anomaly_score"] = 0.0
    evil_mask = df["evil"] == 1
    df.loc[evil_mask, "anomaly_score"] = rng.uniform(0.6, 0.99, size=evil_mask.sum())
    sus_mask = (df["sus"] == 1) & (df["evil"] == 0)
    df.loc[sus_mask, "anomaly_score"] = rng.uniform(0.3, 0.7, size=sus_mask.sum())
    benign_mask = (df["sus"] == 0) & (df["evil"] == 0)
    df.loc[benign_mask, "anomaly_score"] = rng.uniform(0.0, 0.25, size=benign_mask.sum())
    return df


def color_for_score(score):
    if score >= 0.6:
        return "#FF4136"  # red — malicious
    elif score >= 0.3:
        return "#FFB700"  # amber — suspicious
    return "#4A4A4A"  # gray — benign


def event_label(row):
    if row["evil"] == 1:
        return "MALICIOUS"
    if row["sus"] == 1:
        return "SUSPICIOUS"
    return "BENIGN"


# ── Shared sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("SOC Monitor Prototype")
st.sidebar.caption("Throwaway UI exploration — not production code.")

variant = st.sidebar.radio(
    "Select variant",
    ["A — SOC Terminal", "B — Event River (Plotly)", "C — Tactical Grid"],
    key="variant",
)

with st.sidebar.expander("Controls", expanded=True):
    playback_speed = st.slider("Playback speed", 1, 10, 3, key="playback")
    score_threshold = st.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05, key="threshold")
    if st.button("🔀 Reshuffle data", key="reshuffle"):
        st.cache_data.clear()
        st.rerun()

df = generate_event_data()

# ── ═══════════════════════════════════════════════════════════════════════════
# VARIANT A — SOC Terminal
# ═══════════════════════════════════════════════════════════════════════════════
def variant_a():
    st.markdown("""
    <style>
    .stApp { background: #0a0e14; }
    .terminal-header {
        font-family: 'Courier New', monospace;
        color: #00ff88;
        font-size: 14px;
        padding: 10px 0;
        border-bottom: 1px solid #1a3a2a;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    total_events = len(df)
    malicious = int(df["evil"].sum())
    suspicious = int((df["sus"] == 1).sum())
    max_score = float(df["anomaly_score"].max())

    with col1:
        st.metric("Events/sec", f"{random.randint(380, 420)}", delta=f"+{random.randint(5, 20)}")
    with col2:
        st.metric("Malicious Events", malicious, delta=f"{malicious - 180}" if malicious > 180 else f"{malicious}", delta_color="inverse")
    with col3:
        st.metric("Active Threats", random.randint(1, 3), delta="1 new", delta_color="inverse")
    with col4:
        st.metric("Peak Score", f"{max_score:.2f}", delta=f"+{max_score - 0.85:.2f}", delta_color="inverse")

    # Live score gauge
    st.markdown("### Real-time Threat Level")
    gauge_cols = st.columns(3)
    for idx, (label, val) in enumerate([
        ("Current Score", random.uniform(0.6, 0.95)),
        ("30s Average", random.uniform(0.4, 0.7)),
        ("5m Baseline", 0.12),
    ]):
        color = "#FF4136" if val > 0.5 else ("#FFB700" if val > 0.25 else "#2ECC40")
        gauge_html = f"""
        <div style="text-align:center">
            <div style="color:#888;font-size:12px;margin-bottom:4px">{label}</div>
            <div style="
                position:relative;
                width:120px;height:120px;margin:0 auto;
                border-radius:50%;
                background: conic-gradient({color} 0deg {int(val*360)}deg, #1a1a2e {int(val*360)}deg 360deg);
            ">
                <div style="
                    position:absolute;top:50%;left:50%;
                    transform:translate(-50%,-50%);
                    width:90px;height:90px;border-radius:50%;
                    background:#0a0e14;
                    display:flex;align-items:center;justify-content:center;
                    font-size:28px;font-weight:bold;font-family:monospace;
                    color:{color};
                ">{val:.2f}</div>
            </div>
        </div>
        """
        with gauge_cols[idx]:
            components.html(gauge_html, height=140)

    # Attack log feed
    st.markdown("### Threat Event Log")
    log_placeholder = st.empty()

    evil_df = df[df["evil"] == 1].head(30)
    log_lines = []
    for _, row in evil_df.iterrows():
        ts = row["timestamp"].strftime("%H:%M:%S.%f")[:10]
        proc = row["processName"]
        score = row["anomaly_score"]
        color = "#FF4136" if score > 0.8 else "#FFB700"
        log_lines.append(
            f'<div style="font-family:Courier New,monospace;font-size:13px;padding:3px 0;border-bottom:1px solid #1a1a2e">'
            f'<span style="color:#666">[{ts}]</span> '
            f'<span style="color:{color}">●</span> '
            f'<span style="color:#ddd">event={row["eventId"]} user={row["userId"]}</span> '
            f'<span style="color:#888">proc=</span><span style="color:{color}">{proc}</span> '
            f'<span style="color:#888">score=</span><span style="color:{color}">{score:.3f}</span>'
            f'<span style="color:#FF4136;margin-left:10px">⚠ MALICIOUS</span>' if score > 0.8 else
            f'<span style="color:#FFB700;margin-left:10px">◇ suspicious</span>'
            f'</div>'
        )

    log_placeholder.markdown(
        '<div style="background:#0d1117;border:1px solid #1a3a2a;border-radius:4px;padding:10px;max-height:400px;overflow-y:auto">'
        + "\n".join(log_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    # MITRE ATT&CK coverage bar
    st.markdown("### Threat Coverage — MITRE ATT&CK")
    tactic_cols = st.columns(len(MITRE_TACTICS[:6]))
    for i, tactic in enumerate(MITRE_TACTICS[:6]):
        fill = random.uniform(0, 1)
        color = "#FF4136" if fill > 0.6 else ("#FFB700" if fill > 0.2 else "#2ECC40")
        with tactic_cols[i]:
            st.markdown(f"""
            <div style="text-align:center">
                <div style="
                    width:100%;height:80px;background:#1a1a2e;border-radius:4px;
                    position:relative;overflow:hidden;
                ">
                    <div style="
                        position:absolute;bottom:0;width:100%;
                        height:{int(fill*100)}%;background:{color};
                        border-radius:0 0 4px 4px;opacity:0.7;
                    "></div>
                    <div style="
                        position:absolute;top:50%;left:50%;
                        transform:translate(-50%,-50%);
                        color:{color};font-size:11px;font-weight:bold;
                        text-align:center;line-height:1.2;
                    ">{tactic}<br>{fill:.0%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── ═══════════════════════════════════════════════════════════════════════════
# VARIANT B — Event River (Plotly animated timeline)
# ═══════════════════════════════════════════════════════════════════════════════
def variant_b():
    st.markdown("### Event River — Real-time Kernel Process Monitor")

    # Subset for performance
    plot_df = df.tail(800).copy()

    # Main timeline
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.65, 0.15, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Event Stream (color = severity)", "Attack Density", "Anomaly Score Over Time"),
    )

    colors = [color_for_score(s) for s in plot_df["anomaly_score"]]
    labels = [event_label(r) for _, r in plot_df.iterrows()]

    # Row 1: event scatter
    for label_name, label_color in [("BENIGN", "#4A4A4A"), ("SUSPICIOUS", "#FFB700"), ("MALICIOUS", "#FF4136")]:
        mask = [l == label_name for l in labels]
        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"][mask],
                y=[1] * mask.count(True),
                mode="markers",
                name=label_name,
                marker=dict(
                    color=label_color,
                    size=10 if label_name == "MALICIOUS" else (7 if label_name == "SUSPICIOUS" else 4),
                    symbol="diamond" if label_name == "MALICIOUS" else ("triangle-up" if label_name == "SUSPICIOUS" else "circle"),
                    line=dict(width=1 if label_name == "MALICIOUS" else 0, color="white"),
                ),
                hovertemplate=f"<b>{label_name}</b><br>%{{x}}<br>proc: %{{customdata}}<extra></extra>",
                customdata=plot_df["processName"],
            ),
            row=1, col=1,
        )

    # Row 2: attack density histogram
    evil_times = plot_df[plot_df["evil"] == 1]["timestamp"]
    if len(evil_times) > 0:
        fig.add_trace(
            go.Histogram(
                x=evil_times,
                nbinsx=40,
                marker_color="#FF4136",
                opacity=0.7,
                name="Attack Events",
                hovertemplate="Attack events in window: %{y}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Row 3: rolling anomaly score
    window = max(5, len(plot_df) // 100)
    rolling_score = plot_df["anomaly_score"].rolling(window=window, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=rolling_score,
            mode="lines",
            name="Rolling Score",
            line=dict(color="#FFB700", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 183, 0, 0.1)",
            hovertemplate="Score: %{y:.3f}<extra></extra>",
        ),
        row=3, col=1,
    )
    # Threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="#FF4136", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0d1117",
        font=dict(color="#aaa"),
        margin=dict(l=40, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1, gridcolor="#1a1a2e")
    fig.update_yaxes(title_text="Score", row=3, col=1, range=[0, 1], gridcolor="#1a1a2e")
    fig.update_xaxes(title_text="Time", row=3, col=1, gridcolor="#1a1a2e")

    st.plotly_chart(fig, use_container_width=True, key="variant_b_chart")

    # Top alerts panel
    st.markdown("### Active Alerts")
    alert_df = df[df["evil"] == 1].nlargest(6, "anomaly_score")
    alert_cols = st.columns(len(alert_df))
    for i, (_, row) in enumerate(alert_df.iterrows()):
        with alert_cols[i]:
            score = row["anomaly_score"]
            border = "#FF4136" if score > 0.8 else "#FFB700"
            components.html(f"""
            <div style="
                background:#0d1117;border:2px solid {border};border-radius:8px;
                padding:12px;text-align:center;font-family:monospace;min-height:140px;
            ">
                <div style="color:{border};font-size:24px;font-weight:bold">{score:.3f}</div>
                <div style="color:#ccc;font-size:12px;margin-top:4px">{row['processName']}</div>
                <div style="color:#666;font-size:10px;margin-top:4px">{row['timestamp'].strftime('%H:%M:%S')}</div>
                <div style="
                    margin-top:8px;padding:4px 8px;border-radius:4px;
                    background:{border};color:#000;font-size:10px;font-weight:bold;
                    display:inline-block;
                ">ALERT</div>
            </div>
            """, height=200)


# ── ═══════════════════════════════════════════════════════════════════════════
# VARIANT C — Tactical Grid
# ═══════════════════════════════════════════════════════════════════════════════
def variant_c():
    # Row 1: KPI tiles
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    total_events = len(df)
    malicious_count = int(df["evil"].sum())
    suspicious_count = int(df["sus"].sum())
    benign_count = total_events - malicious_count - suspicious_count

    with k1:
        st.metric("Total Events", f"{total_events:,}", delta=f"+{random.randint(50,200)}")
    with k2:
        st.metric("🔴 Malicious", f"{malicious_count:,}", delta="+23", delta_color="inverse")
    with k3:
        st.metric("🟡 Suspicious", f"{suspicious_count:,}", delta="-5", delta_color="normal")
    with k4:
        st.metric("True Positive Rate", f"{random.uniform(0.91, 0.97):.1%}", delta="+2.1%")
    with k5:
        st.metric("False Positive Rate", f"{random.uniform(0.01, 0.04):.1%}", delta="-0.5%", delta_color="inverse")
    with k6:
        st.metric("Mean Time to Detect", f"{random.uniform(2, 8):.1f}s", delta="-1.2s", delta_color="inverse")

    # Row 2: MITRE ATT&CK heatmap + Process tree
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### MITRE ATT&CK Coverage Heatmap")

        # Generate random tactic×technique scores
        heatmap_data = {}
        for tactic in MITRE_TACTICS:
            heatmap_data[tactic] = {t: random.uniform(0, 1) if random.random() < 0.4 else 0
                                     for t in [f"T{random.randint(1000,1600)}" for _ in range(5)]}

        # Build a proper matrix with real technique IDs
        tactic_list = []
        technique_list = []
        values = []
        for tactic in MITRE_TACTICS:
            techs = MITRE_TECHNIQUES.get(tactic, ["T1000", "T1001", "T1002"])
            for tech in techs:
                tactic_list.append(tactic)
                technique_list.append(tech)
                hit = random.random()
                values.append(hit if hit < 0.35 else 0.0)

        heatmap_df = pd.DataFrame({
            "Tactic": tactic_list,
            "Technique": technique_list,
            "Coverage": values,
        })

        fig_hm = px.density_heatmap(
            heatmap_df,
            x="Technique", y="Tactic",
            z="Coverage",
            color_continuous_scale=["#0d1117", "#1a3a2a", "#2ECC40", "#FFB700", "#FF4136"],
            range_color=[0, 1],
            height=400,
        )
        fig_hm.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e14",
            plot_bgcolor="#0d1117",
            font=dict(color="#aaa", size=10),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title="Hit", thickness=10, len=0.5),
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="variant_c_heatmap")

    with right:
        st.markdown("#### Process Relationship Graph")
        # Force-directed graph using D3 via components.html
        nodes_json = [
            {"id": "sshd", "group": 1, "count": random.randint(200, 400)},
            {"id": "bash", "group": 1, "count": random.randint(300, 600)},
            {"id": "systemd", "group": 1, "count": random.randint(100, 300)},
            {"id": "wget", "group": 2, "count": random.randint(20, 50)},
            {"id": "curl", "group": 2, "count": random.randint(10, 40)},
            {"id": "chmod", "group": 2, "count": random.randint(5, 20)},
            {"id": "xmrig", "group": 3, "count": random.randint(10, 30)},
            {"id": "reverse-shell", "group": 3, "count": random.randint(3, 10)},
            {"id": "data-exfil", "group": 3, "count": random.randint(2, 8)},
        ]
        links_json = [
            {"source": "sshd", "target": "bash", "value": 80},
            {"source": "bash", "target": "wget", "value": 25},
            {"source": "bash", "target": "curl", "value": 18},
            {"source": "wget", "target": "chmod", "value": 10},
            {"source": "curl", "target": "chmod", "value": 8},
            {"source": "chmod", "target": "xmrig", "value": 15},
            {"source": "xmrig", "target": "reverse-shell", "value": 5},
            {"source": "reverse-shell", "target": "data-exfil", "value": 4},
        ]

        d3_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{ margin:0; background:#0d1117; overflow:hidden; font-family:monospace; }}
            .link {{ stroke: #333; stroke-opacity: 0.6; }}
            .node circle {{ stroke: #fff; stroke-width: 1.5px; }}
            .node text {{ fill: #ccc; font-size: 11px; font-family: monospace; }}
            .tooltip {{ position: absolute; background: #1a1a2e; color: #ccc; padding: 4px 8px;
                        border-radius: 4px; font-size: 11px; pointer-events: none; border: 1px solid #333; }}
        </style>
        </head>
        <body>
        <svg width="100%" height="100%"></svg>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
        const data = {{
            nodes: {nodes_json},
            links: {links_json}
        }};
        const width = window.innerWidth * 0.48;
        const height = 400;
        const color = d3.scaleOrdinal()
            .domain([1, 2, 3])
            .range(["#2ECC40", "#FFB700", "#FF4136"]);

        const svg = d3.select("svg")
            .attr("width", width)
            .attr("height", height);

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const node = svg.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
                .on("drag", (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
                .on("end", (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

        node.append("circle")
            .attr("r", d => Math.sqrt(d.count) * 1.2 + 5)
            .attr("fill", d => color(d.group));

        node.append("text")
            .attr("dy", d => -Math.sqrt(d.count) * 1.2 - 8)
            .attr("text-anchor", "middle")
            .text(d => d.id);

        // Pulsing effect on red nodes (malicious)
        function pulse() {{
            node.selectAll("circle")
                .filter(d => d.group === 3)
                .transition()
                .duration(800)
                .attr("r", d => Math.sqrt(d.count) * 1.2 + 8)
                .transition()
                .duration(800)
                .attr("r", d => Math.sqrt(d.count) * 1.2 + 5)
                .on("end", pulse);
        }}
        pulse();

        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
        </script>
        </body>
        </html>
        """
        components.html(d3_html, height=420)

    st.markdown("#### Event Timeline — Animated Playback")

    # Attack replay — animated scatter over time
    replay_df = df.iloc[800:1050].copy()
    frames = []
    chunk_size = 5

    for i in range(0, len(replay_df), chunk_size):
        chunk = replay_df.iloc[:i + chunk_size]
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=chunk["timestamp_numeric"],
                    y=chunk["anomaly_score"],
                    mode="markers+lines",
                    marker=dict(
                        color=[color_for_score(s) for s in chunk["anomaly_score"]],
                        size=[12 if e == 1 else (8 if s == 1 else 5)
                               for e, s in zip(chunk["evil"], chunk["sus"])],
                        symbol=["diamond" if e == 1 else ("triangle-up" if s == 1 else "circle")
                                for e, s in zip(chunk["evil"], chunk["sus"])],
                    ),
                    line=dict(color="#333", width=1),
                    name="Events",
                    hovertemplate="<b>%{text}</b><br>Score: %{y:.3f}<extra></extra>",
                    text=chunk["processName"],
                ),
            ],
            name=f"frame{i}",
        )
        frames.append(frame)

    fig_anim = go.Figure(
        data=[go.Scatter(
            x=[replay_df.iloc[0]["timestamp_numeric"]],
            y=[replay_df.iloc[0]["anomaly_score"]],
            mode="markers",
            marker=dict(color="#4A4A4A", size=5),
            name="Events",
        )],
        frames=frames,
        layout=go.Layout(
            template="plotly_dark",
            height=250,
            paper_bgcolor="#0a0e14",
            plot_bgcolor="#0d1117",
            font=dict(color="#aaa", size=10),
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title="Seconds from boot", range=[
                replay_df["timestamp_numeric"].min() - 5,
                replay_df["timestamp_numeric"].max() + 5,
            ], gridcolor="#1a1a2e"),
            yaxis=dict(title="Anomaly Score", range=[-0.05, 1.05], gridcolor="#1a1a2e"),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 80, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
                x=0.1, y=0,
                xanchor="right", yanchor="bottom",
                bgcolor="#1a1a2e", bordercolor="#333",
            )],
        ),
    )
    fig_anim.add_hline(y=0.5, line_dash="dash", line_color="#FF4136", opacity=0.5)
    st.plotly_chart(fig_anim, use_container_width=True, key="variant_b_anim")

    # Bottom panels
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### Top Processes by Alert Count")
        proc_counts = df[df["evil"] == 1]["processName"].value_counts().head(8)
        fig_bar = go.Figure(go.Bar(
            x=proc_counts.values,
            y=proc_counts.index,
            orientation="h",
            marker=dict(
                color=proc_counts.values,
                colorscale=[[0, "#FFB700"], [1, "#FF4136"]],
            ),
            text=proc_counts.values,
            textposition="outside",
        ))
        fig_bar.update_layout(
            template="plotly_dark", height=250, margin=dict(l=10, r=30, t=10, b=10),
            paper_bgcolor="#0a0e14", plot_bgcolor="#0d1117", font=dict(color="#aaa", size=10),
            xaxis=dict(title="Alert Count", gridcolor="#1a1a2e"),
            yaxis=dict(gridcolor="#1a1a2e"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="variant_b_bars")

    with b2:
        st.markdown("#### User Activity Breakdown")
        user_data = df.groupby("userId").agg(
            total=("anomaly_score", "count"),
            avg_score=("anomaly_score", "mean"),
            malicious=("evil", "sum"),
        ).reset_index()
        user_data["risk"] = user_data["avg_score"].apply(
            lambda x: "🔴 High" if x > 0.5 else ("🟡 Med" if x > 0.2 else "🟢 Low")
        )
        fig_scatter = go.Figure()
        for risk, color, sym in [("🟢 Low", "#2ECC40", "circle"), ("🟡 Med", "#FFB700", "triangle-up"), ("🔴 High", "#FF4136", "diamond")]:
            mask = user_data["risk"] == risk
            fig_scatter.add_trace(go.Scatter(
                x=user_data[mask]["total"],
                y=user_data[mask]["avg_score"],
                mode="markers+text",
                name=risk,
                text=[f"uid={int(u)}" for u in user_data[mask]["userId"]],
                textposition="top center",
                marker=dict(color=color, size=user_data[mask]["malicious"] * 15 + 10, symbol=sym),
                hovertemplate="User %{text}<br>Events: %{x}<br>Score: %{y:.3f}<extra></extra>",
            ))
        fig_scatter.update_layout(
            template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0a0e14", plot_bgcolor="#0d1117", font=dict(color="#aaa", size=10),
            xaxis=dict(title="Total Events", type="log", gridcolor="#1a1a2e"),
            yaxis=dict(title="Avg Score", range=[0, 1], gridcolor="#1a1a2e"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="variant_b_scatter")


# ── Router ───────────────────────────────────────────────────────────────────
st.title("🛡️ Sentinel — SOC Monitor")
st.caption(f"Prototype dashboard — {variant.split(' —')[0]} variant. Throwaway code; do not ship.")
st.divider()

if "A" in variant:
    variant_a()
elif "B" in variant:
    variant_b()
elif "C" in variant:
    variant_c()


# ── Floating variant notes ───────────────────────────────────────────────────
st.divider()
with st.expander("💡 What's different between variants", expanded=False):
    st.markdown("""
    | Variant | Style | Best for |
    |---------|-------|----------|
    | **A — SOC Terminal** | Dark terminal, live log feed, circular gauges, tactical bar charts | Cinematic feel, "war room" aesthetic |
    | **B — Event River** | Plotly scatter timeline + histogram + rolling score line, alert cards | Data-dense analysis, seeing temporal patterns |
    | **C — Tactical Grid** | Multi-panel: KPI tiles, MITRE heatmap, D3 process graph, animated replay, bar + scatter charts | Defense briefings, showing breadth |
    """)

st.caption("PROTOTYPE — fold winner into real project and delete this file.")
