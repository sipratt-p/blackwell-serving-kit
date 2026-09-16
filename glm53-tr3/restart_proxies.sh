#!/bin/bash
# thinking-budget proxy: public :8012 -> vLLM :8022
p=$(ss -ltnp 2>/dev/null | grep ":8012 " | grep -o 'pid=[0-9]*' | cut -d= -f2); [ -n "$p" ] && kill $p; sleep 1
cd "$(dirname "$0")"
setsid nohup env GLM_THINK_BUDGET=${GLM_THINK_BUDGET:-8192} UPSTREAM=http://127.0.0.1:8022 PORT=8012 python3 think_budget_proxy.py >> proxy.log 2>&1 < /dev/null &
sleep 1; ss -ltn | grep -c ":8012 "
