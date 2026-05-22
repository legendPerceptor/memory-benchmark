#!/bin/bash
# Memory Benchmark - One-click benchmark script
# Usage: ./scripts/benchmark.sh [--system mempalace] [--benchmark locomo] [--top-k 10]

set -e

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 默认参数
SYSTEMS="mempalace"
BENCHMARK="locomo"
TOP_K=10
GRANULARITY="session"
MODE="raw"
OUTPUT="reports"
FORMAT="both"

# 数据集路径
DATA_PATH="../../memory-eval/locomo/data/locomo10.json"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --system|--systems)
            SYSTEMS="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --granularity)
            GRANULARITY="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --system <name>     Memory system to test (mempalace,ogmemory,memsearch)"
            echo "                      Use comma-separated list for multiple systems"
            echo "  --benchmark <name>  Benchmark to run (locomo)"
            echo "  --top-k <n>        Number of results to retrieve (default: 10)"
            echo "  --granularity <g>  Corpus granularity (session|dialog, default: session)"
            echo "  --mode <m>         Retrieval mode for MemPalace (raw|hybrid|aaak)"
            echo "  --output <dir>     Output directory (default: reports)"
            echo "  --format <f>       Output format (json|markdown|both, default: both)"
            echo "  --data <path>      Path to dataset"
            echo ""
            echo "Examples:"
            echo "  $0 --system mempalace"
            echo "  $0 --systems mempalace,ogmemory --top-k 5"
            echo "  $0 --system mempalace --mode hybrid --granularity dialog"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT"

echo "========================================"
echo "  Memory System Benchmark"
echo "========================================"
echo "  Systems: $SYSTEMS"
echo "  Benchmark: $BENCHMARK"
echo "  Top-K: $TOP_K"
echo "  Granularity: $GRANULARITY"
echo "  Mode: $MODE"
echo "  Output: $OUTPUT"
echo "========================================"

# 运行 Python 脚本
cd src
python3 runner.py \
    --systems "$SYSTEMS" \
    --benchmark "$BENCHMARK" \
    --data "$DATA_PATH" \
    --top-k "$TOP_K" \
    --granularity "$GRANULARITY" \
    --mode "$MODE" \
    --output "../$OUTPUT" \
    --format "$FORMAT"

echo ""
echo "Done!"