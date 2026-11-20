#!/bin/bash
echo "=== PROMPT ($1, first part) ==="
jq -r '.messages[0].content' ${1}_claim.json | fold -s -w 70 | head -n 22
echo "... [prompt continues, about 2,000 characters in total]"
echo
echo ">>> Sending $1 through LiteLLM (model: auto)"
curl -s -m 240 http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $DEMO_KEY" -H "Content-Type: application/json" \
  -d @${1}_claim.json > /tmp/resp.json
echo
echo "=== ROUTER LOG ==="
vllm-sr logs router 2>/dev/null | grep -E 'hard_score|routing_decision' | tail -n 2 \
  | grep -oE 'hard_score=[^"]*|"decision":"[^"]*"|"selected_model":"[^"]*"'
echo
echo "=== RESPONSE ==="
jq -r '.choices[0].message.content // .error.message' /tmp/resp.json | fold -s -w 70 | head -n 25
