"""Conservative, manually reviewed grouping of Password-labeled schema fields.

The labels describe the parameter text, not observed runtime collection.
Unknown and generic descriptions stay unknown; no purpose is inferred from an
Action identifier or from the parameter name alone.
"""

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PURPOSES = {
    ("password", "Password for SMPP account. This is required to send SMS. Obtain from members.telecomsxchange.com if you do not have one."): "SMPP messaging",
    ("password", "The database's user password"): "Database access",
    ("password", "The MySQL password"): "Database access",
    ("password", "The PostgreSQL password"): "Database access",
    ("password", "The  SQL password"): "Database access",
    ("setPassword", "On editting set the authentication password for connection to the host."): "Host connection",
    ("password", "A password to protect access to the content and excerpt."): "Content or resume protection",
    ("password", "The password for the post if it is password protected."): "Content or resume protection",
    ("password", "Password for the resume."): "Content or resume protection",
    ("password", "The password of the user where the article will be posted. You should always ask the person for the password and they should inform you in chat"): "Article-posting account",
    ("password", ""): "General or unclear",
    ("api_password", ""): "General or unclear",
    ("password", "The password of the user"): "General or unclear",
    ("password", "Password for authentication."): "General or unclear",
    ("password", "User's password"): "General or unclear",
    # The description says "Username" although the field is named password.
    ("password", "Username used for authenticatin the service on the host."): "General or unclear",
    ("password", "Password of the user"): "General or unclear",
    ("passcode", "The passcode associated with the user's account for authentication."): "General or unclear",
}

DISPLAY_ORDER = [
    "Database access",
    "Content or resume protection",
    "SMPP messaging",
    "Host connection",
    "Article-posting account",
    "General or unclear",
]


def summarize_password_use_cases(records):
    password_rows = [row for row in records if row["data_type"] == "Password"]
    counts = Counter()
    for row in password_rows:
        key = (row["name"], row.get("description") or "")
        if key not in PURPOSES:
            raise ValueError(f"Unreviewed Password field: {key!r}")
        counts[PURPOSES[key]] += 1
    action_ids = {
        action_id
        for row in password_rows
        for action_id in row["plugin_id_filenames"]
    }
    return {
        "record_count": len(password_rows),
        "action_count": len(action_ids),
        "group_counts": {label: counts[label] for label in DISPLAY_ORDER},
    }


def plot_password_use_cases(summary):
    counts = summary["group_counts"]
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    fig.patch.set_facecolor("#fbfbfa")
    ax.set_facecolor("#fbfbfa")
    labels = DISPLAY_ORDER[::-1]
    values = [counts[label] for label in labels]
    colors = ["#a7a7a1" if label == "General or unclear" else "#b4232e" for label in labels]
    bars = ax.barh(labels, values, color=colors, height=0.65)
    for bar, value in zip(bars, values):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(values) + 1.4)
    ax.set_xlabel("Password-labeled records (not Actions)", fontsize=11)
    ax.set_title("Where do password fields appear?", loc="left", fontsize=16, fontweight="bold", pad=30)
    ax.text(0, 1.04, f"{summary['record_count']} records across {summary['action_count']} distinct Actions", transform=ax.transAxes, fontsize=12, color="#b4232e", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.tick_params(axis="x", colors="#666666")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.03, "Purpose coded from schema descriptions; generic or inconsistent text stays unclear.\nNo runtime password transfer is shown.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / "backend/data/data_entries_final.json").read_text(encoding="utf-8"))
    summary = summarize_password_use_cases(records)
    figure = plot_password_use_cases(summary)
    for folder in (root / "notebooks/exported_charts", root / "backend/static/charts"):
        figure.savefig(folder / "password_use_cases.png", dpi=150, bbox_inches="tight")
    print(summary)
