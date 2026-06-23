#!/usr/bin/env bash
set -euo pipefail

# Check if output directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <output_directory>"
    echo "Example: $0 /root/codex_baseline"
    exit 1
fi

OUTPUT_DIR="$1"
OUTPUT_FILE="${OUTPUT_DIR}/system_info.txt"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Collecting system information..."
echo "Output will be saved to: $OUTPUT_FILE"
echo

# Redirect all output to both stdout and file
exec > >(tee "$OUTPUT_FILE")

echo "=== SYSTEM BASELINE ==="
uname -a || true
cat /etc/os-release || true
ldd --version | head -1 || true
echo

echo "=== FIRMWARE & MICROCODE ==="
sudo dmidecode -t bios || true
dmesg | grep -i microcode || true
echo

echo "=== CPU ==="
lscpu -e || true
lscpu || true
grep -m1 "model name" /proc/cpuinfo || true
lscpu --cache || true
command -v cpupower >/dev/null && cpupower frequency-info || true
[ -f /sys/devices/system/cpu/smt/active ] && cat /sys/devices/system/cpu/smt/active || true
command -v turbostat >/dev/null && sudo turbostat --Summary --quiet --interval 1 --iterations 1 || true
echo

echo "=== MEMORY & NUMA ==="
numactl --hardware || true
numastat -m || true
command -v lstopo >/dev/null && lstopo --no-io --no-graphics || true
sudo dmidecode -t memory | egrep "Size:|Speed:|Type:|Configured|Total Width|Data Width|Error Correction" || true
grep -H . /sys/kernel/mm/transparent_hugepage/* || true
sysctl vm.nr_hugepages || true
echo

echo "=== PCIe / TOPOLOGY ==="
lspci -nn | egrep -i "nvidia|amd|advanced micro|intel corporation|mlx|infiniband|nvme" || true
lspci -vv | egrep -i "LnkCap|LnkSta" -A2 | sed 's/^[[:space:]]*//' || true
command -v hwloc-ls >/dev/null && hwloc-ls --whole-io || true
echo

echo "=== NVIDIA (if present) ==="
command -v nvidia-smi >/dev/null && nvidia-smi || true
command -v nvidia-smi >/dev/null && nvidia-smi -q || true
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,cuda_version,pci.bus_id,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv || true
command -v nvidia-smi >/dev/null && nvidia-smi topo -m || true
echo

echo "=== AMD ROCm (if present) ==="
command -v rocminfo >/dev/null && rocminfo || true
command -v rocm-smi >/dev/null && rocm-smi --showdriverversion --showproductname --showbus --showpcie --showclocks --showmaxclocks || true
echo

echo "=== Intel GPU (if present) ==="
command -v sycl-ls >/dev/null && sycl-ls || true
command -v ze_device_queries >/dev/null && ze_device_queries || true
command -v intel_gpu_top >/dev/null && intel_gpu_top -L || true
echo

echo "=== COMPILERS & OPENMP RUNTIMES ==="
which gcc clang icx nvcc 2>/dev/null || true
gcc --version 2>/dev/null | head -1 || true
clang --version 2>/dev/null | head -1 || true
icx --version 2>/dev/null || true
nvcc --version 2>/dev/null || true
echo "libomptarget plugins (clang):"
clang -print-resource-dir 2>/dev/null | xargs -I{} bash -c 'ls {}/lib//libomptarget.rtl. 2>/dev/null' || true
echo

echo "=== OS & SCHEDULING KNOBS ==="
sysctl kernel.numa_balancing || true
systemctl is-active irqbalance && systemctl status irqbalance --no-pager || true
cat /proc/cmdline || true
cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null || true
free -h || true
swapon --show || true
sysctl vm.swappiness || true
echo

echo "=== STORAGE (optional) ==="
lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT || true
command -v nvme >/dev/null && nvme list || true

echo "=== DONE ==="
echo
echo "System information collection complete!"
echo "Output saved to: $OUTPUT_FILE"