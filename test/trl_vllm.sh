IMG_B64=$(base64 -w0 /mnt/task_runtime/Med-R1/Images/xxxx.png)

cat > /tmp/vllm_req.json <<EOF
{
  "prompts": ["Describe the main findings in the image."],
  "images": ["$IMG_B64"],
  "max_tokens": 64
}
EOF

curl -s http://127.0.0.1:8001/generate/ \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/vllm_req.json