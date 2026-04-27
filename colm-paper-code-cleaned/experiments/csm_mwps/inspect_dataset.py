import json
import os

def inspect(path, name):
    with open(path) as f:
        d = json.load(f)

    print("=" * 80)
    print(f"  DATASET: {name}")
    print(f"  Total MWPs in file: {len(d)}")
    print("=" * 80)

    # Summary stats first
    total_distractors = []
    for mwp_id, mwp in d.items():
        for inst_id, inst in mwp["instantiations"].items():
            n = len(inst.get("misconception_answers", []))
            total_distractors.append(n)

    print(f"\n--- SUMMARY ---")
    print(f"  Total instantiations: {len(total_distractors)}")
    print(f"  Distractors per instantiation: min={min(total_distractors)}, max={max(total_distractors)}, avg={sum(total_distractors)/len(total_distractors):.1f}")
    print()

    # Print first MWP in detail (all instantiations)
    k = list(d.keys())[0]
    mwp = d[k]

    print(f"--- DETAILED VIEW: MWP ID = {k} ---")
    print(f"  # instantiations: {len(mwp['instantiations'])}")
    print(f"\n  METADATA:")
    print(f"  {json.dumps(mwp.get('metadata', {}), indent=4)}")

    for inst_id in sorted(mwp["instantiations"].keys()):
        inst = mwp["instantiations"][inst_id]
        print(f"\n  {'─' * 60}")
        print(f"  INSTANTIATION {inst_id}")
        print(f"  {'─' * 60}")

        print(f"\n  PROBLEM (inconsistent):")
        print(f"    {inst.get('problem', 'N/A')}")

        print(f"\n  PROBLEM (consistent):")
        print(f"    {inst.get('cons_problem', 'N/A')}")

        ca = inst.get("correct_answer", {})
        print(f"\n  CORRECT ANSWER: {ca.get('answer')}")
        if ca.get("rt"):
            print(f"  CORRECT REASONING TRACE:")
            for line in ca["rt"].split(". "):
                if line.strip():
                    print(f"    {line.strip()}")

        misconceptions = inst.get("misconception_answers", [])
        print(f"\n  MISCONCEPTION DISTRACTORS: {len(misconceptions)} total")
        for i, ma in enumerate(misconceptions):
            print(f"\n    [{i}] Answer: {ma.get('answer')}  |  Plausible: {ma.get('plausible')}")
            for key in ma:
                if key not in ("answer", "plausible", "rt"):
                    print(f"        {key}: {ma[key]}")
            if ma.get("rt"):
                print(f"        Reasoning trace:")
                for line in ma["rt"].split(". "):
                    if line.strip():
                        print(f"          {line.strip()}")

    print("\n")


if __name__ == "__main__":
    files = [
        ("/tmp/test_ci.json", "CI (comparison/keyword)"),
    ]

    if os.path.exists("/tmp/test_am.json"):
        files.append(("/tmp/test_am.json", "AM (arithmetic)"))

    for path, name in files:
        if os.path.exists(path):
            inspect(path, name)
        else:
            print(f"File not found: {path}")