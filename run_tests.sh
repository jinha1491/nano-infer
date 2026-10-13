set -e
for t in mlp attn model correct batch continuous speculative; do
    echo "=== test_$t ==="
    python tests/test_$t.py
done
echo "ALL TESTS PASSED"
