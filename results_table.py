"""
Build the master results CSV and a quick-glance markdown table.

Reads from results.json. Run once after every set of experiments.
"""

import csv
import json
import os
import sys


HERE = os.path.dirname(__file__)


def main():
    with open(os.path.join(HERE, "results.json")) as f:
        data = json.load(f)
    pipelines = data["samsum_test_100"]["pipelines"]
    paper_full = data["samsum_test_100"]["promptsum_paper_full_shot"]

    rows = [["pipeline", "owner", "signal", "rouge1", "rouge2", "rougeL", "delta_R1_vs_paper"]]
    for name, m in pipelines.items():
        rows.append([
            name,
            m.get("owner", ""),
            m.get("signal", ""),
            m["rouge1"], m["rouge2"], m["rougeL"],
            round(m["rouge1"] - paper_full["rouge1"], 2),
        ])

    out_csv = os.path.join(HERE, "..", "results", "all_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {out_csv}")

    # Markdown
    md_lines = ["| pipeline | owner | signal | R-1 | R-2 | R-L | \u0394R-1 vs paper |",
                "| -------- | ----- | ------ | --- | --- | --- | -------------- |"]
    for name, m in pipelines.items():
        md_lines.append(f"| {name} | {m.get('owner','')} | {m.get('signal','')} | "
                        f"{m['rouge1']} | {m['rouge2']} | {m['rougeL']} | "
                        f"{round(m['rouge1']-paper_full['rouge1'],2):+.2f} |")
    out_md = os.path.join(HERE, "..", "results", "results_table.md")
    with open(out_md, "w") as f:
        f.write("# SAMSum results — all pipelines\n\n")
        f.write(f"Reference: PromptSum (paper, full-shot): R-1 = {paper_full['rouge1']}, "
                f"R-2 = {paper_full['rouge2']}, R-L = {paper_full['rougeL']}\n\n")
        f.write("\n".join(md_lines))
        f.write("\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
