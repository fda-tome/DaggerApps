#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <manifest.tsv>" >&2
  exit 1
fi

manifest="$1"
[[ -f "$manifest" ]] || { echo "Manifest not found: $manifest" >&2; exit 1; }

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# Expand manifest rows into per-variant mean time rows.
while IFS=$'\t' read -r ts t inner ws rows cols th tw scenario variants outdir; do
  [[ "$ts" == "timestamp" ]] && continue
  csv="$outdir/strong_scaling.csv"
  [[ -f "$csv" ]] || continue
  awk -F, -v t="$t" -v inner="$inner" -v ws="$ws" -v tw="$tw" -v rows="$rows" -v cols="$cols" '
    FNR>1 {
      key=$2
      n[key]++
      s[key]+=$10
    }
    END {
      for (k in n) {
        printf "%s,%s,%s,%s,%s,%s,%.9f\n", k,t,inner,ws,rows,tw,s[k]/n[k]
      }
    }
  ' "$csv" >> "$tmp"
done < "$manifest"

if [[ ! -s "$tmp" ]]; then
  echo "No strong_scaling.csv data found from manifest entries." >&2
  exit 1
fi

echo "variant,thread,inner,wave_size,rows,tile_w,mean_sec,speedup,efficiency"
awk -F, '
  {
    key=$1
    t=$2+0
    mean=$7+0.0
    row[key,t]=$0
    mt[key,t]=mean
    if (!(key in base) || t==1) {
      if (t==1) base[key]=mean
    }
    variants[key]=1
    threads[t]=1
  }
  END {
    for (v in variants) {
      if (!(v in base)) continue
      for (t in threads) {
        if ((v SUBSEP t) in mt) {
          split(row[v,t], a, ",")
          sp=base[v]/mt[v,t]
          eff=sp/t
          printf "%s,%d,%s,%s,%s,%s,%.9f,%.6f,%.6f\n", a[1], t, a[3], a[4], a[5], a[6], a[7], sp, eff
        }
      }
    }
  }
' "$tmp" | sort -t, -k1,1 -k2,2n
