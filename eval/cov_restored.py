import json, glob, collections, sys
E = "/ai4good1-shared/liyu/egoOmni_baselines/eval"
items = json.load(open("/ai4good1-shared/liyu/egoOmni/test/qa_restored_v3.json")); exp = {i["sample_id"] + "#r1" for i in items}
for tag in (sys.argv[1:] or ["gemini_3_8_flash"]):
    rows = [json.loads(l) for f in glob.glob(f"{E}/preds/{tag}/restored_v3/**/shard*.jsonl", recursive=True) for l in open(f) if l.strip()]
    ok = {r["key"] for r in rows if r.get("pred") is not None}
    extra = {r["key"] for r in rows} - exp
    print(f"{tag}: rows={len(rows)} ok={len(ok)} missing={len(exp - ok)} error_rows={sum(r.get('pred') is None for r in rows)} extra_keys={len(extra)} "
          f"modality={dict(collections.Counter(r.get('modality_used') for r in rows))} spend=${sum(r.get('cost_usd') or 0 for r in rows):.2f}")
