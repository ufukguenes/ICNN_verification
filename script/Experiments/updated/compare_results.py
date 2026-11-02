"""
Joins result JSON lines of dhov_rust (results/runs/<config>/<time>/results.jsonl) and of suite.py by (net, spec)
and prints per instance verdict, o-value and total time of every run, a summary per run, and optionally the
per-layer details (unstable neurons after tightening, enlargement shifts) of one instance.

With the default early stop (dhov_rust `[property] stop = "decided"`, suite.py `--stop decided`) compare verdicts
and times: an o-value is then only a sign (empty for unverified instances). o-values are comparable between runs
that solve the class problems to optimality (`stop = "optimal"`, `--stop optimal`) without a pre-check verdict;
instances verified by the pre-check are marked with *.

Example:
  .venv/bin/python script/Experiments/updated/compare_results.py \\
      ../dhov_rust/results/runs/compare_snr_milp_alpha_crown/<time>/results.jsonl \\
      results/python/snr_milp_alpha_crown.jsonl --layers prop_02_0.026.vnnlib
"""
import argparse
import json
import os
import statistics


def load(path):
    runs = {}
    config = None
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                config = config or record.get("config")
                # file names only: dhov_rust writes the spec as given in instances.csv, which may be a path
                runs[(os.path.basename(record["net"]), os.path.basename(record["spec"]))] = record
    return config or path, runs


def o_text(record):
    if record is None:
        return "-"
    value = record.get("o_value")
    mark = "*" if (record.get("precheck") or {}).get("verified") else ""
    return "{}{}".format("-" if value is None else "{:.5g}".format(value), mark)


def total(record):
    return (record or {}).get("times", {}).get("total")


def layer_summary(record):
    rows = []
    for layer in record.get("layers") or []:
        groups = (layer.get("dhov") or {}).get("groups") or []
        enlargements = [g["enlargement"] for g in groups if g.get("enlargement")]
        shifts = [e["shift"] for e in enlargements]
        limited = sum(1 for e in enlargements if e.get("status") != "optimal")
        rows.append("l{} {} unstable {}{}".format(
            layer["layer"], layer.get("strategy"), layer.get("unstable"),
            "" if not shifts else ", {} groups, shift sum {:.4g} (median {:.4g}), {} not optimal".format(
                len(shifts), sum(shifts), statistics.median(shifts), limited)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="results.jsonl files (dhov_rust or suite.py)")
    parser.add_argument("--layers", nargs="*", default=[], help="spec file names to show per-layer details for")
    args = parser.parse_args()

    runs = [load(path) for path in args.results]
    keys = sorted({key for _, records in runs for key in records})
    names = [name for name, _ in runs]
    width = max(14, max(len(n) for n in names))

    print("{:<28}".format("spec") + "".join("{:>{w}}".format(n[:width], w=width + 2) for n in names))
    for key in keys:
        cells = []
        for _, records in runs:
            r = records.get(key)
            t = total(r)
            cells.append("{} {} {}".format((r or {}).get("verdict", "-")[:4], o_text(r),
                                           "-" if t is None else "{:.0f}s".format(t)))
        print("{:<28}".format(key[1][:28]) + "".join("{:>{w}}".format(c, w=width + 2) for c in cells))

    print()
    first_name, first = runs[0]
    for name, records in runs:
        verdicts = [r.get("verdict") for r in records.values()]
        times = [total(r) for r in records.values() if total(r) is not None]
        comparable = [r["o_value"] for r in records.values()
                      if r.get("o_value") is not None and not (r.get("precheck") or {}).get("verified")]
        line = "{}: {} instances, verified {}, falsified {}, unknown {}, timeout {}, error {}; mean time {}; " \
               "mean o-value (no pre-check verdict) {}".format(
                   name, len(records), verdicts.count("verified"), verdicts.count("falsified"),
                   verdicts.count("unknown"), verdicts.count("timeout"), verdicts.count("error"),
                   "-" if not times else "{:.1f}s".format(sum(times) / len(times)),
                   "-" if not comparable else "{:.4g} over {}".format(sum(comparable) / len(comparable),
                                                                      len(comparable)))
        if records is not first:
            # only o-values of the class problems; with stop = "decided" they are mostly empty or only signs
            diffs = [records[k]["o_value"] - first[k]["o_value"] for k in records
                     if k in first and records[k].get("o_value") is not None
                     and first[k].get("o_value") is not None
                     and not (records[k].get("precheck") or {}).get("verified")
                     and not (first[k].get("precheck") or {}).get("verified")]
            same_verdict = sum(1 for k in records if k in first and records[k].get("verdict") == first[k].get("verdict"))
            if diffs:
                line += "; vs {}: same verdict {}/{}, o-value difference mean {:+.4g}, max |.| {:.4g}".format(
                    first_name, same_verdict, len([k for k in records if k in first]),
                    sum(diffs) / len(diffs), max(abs(d) for d in diffs))
        print(line)

    for spec in args.layers:
        print("\n" + spec)
        for name, records in runs:
            record = next((r for (net, s), r in records.items() if s == spec), None)
            print("  " + name + ("" if record else ": -"))
            for row in layer_summary(record or {}):
                print("    " + row)


if __name__ == "__main__":
    main()
