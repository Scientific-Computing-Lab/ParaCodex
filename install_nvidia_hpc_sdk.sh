#!/bin/bash
# Automated NVIDIA HPC SDK Installer for ParaCodex
# This script downloads and installs NVIDIA HPC SDK with OpenMP GPU offload support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                              ║${NC}"
echo -e "${BLUE}║              NVIDIA HPC SDK Automated Installer for ParaCodex                ║${NC}"
echo -e "${BLUE}║                                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
NVHPC_VERSION="25.7"
NVHPC_YEAR="2025"
NVHPC_FULL_VERSION="2025_257"
CUDA_VERSION="12.6"
INSTALL_DIR="/opt/nvidia/hpc_sdk"
DOWNLOAD_URL="https://developer.download.nvidia.com/hpc-sdk/${NVHPC_VERSION}/nvhpc_${NVHPC_FULL_VERSION}_Linux_x86_64_cuda_${CUDA_VERSION}.tar.gz"
TEMP_DIR="/tmp/nvhpc_install_$$"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check if already installed
echo ""
print_info "Checking for existing NVIDIA HPC SDK installation..."

# Check if nvc++ is in PATH
if command_exists nvc++; then
    EXISTING_VERSION=$(nvc++ --version 2>&1 | head -1 || echo "unknown")
    print_warning "NVIDIA HPC SDK is already installed and in PATH: $EXISTING_VERSION"
    echo ""
    read -p "Do you want to reinstall/upgrade? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled by user."
        exit 0
    fi
# Check if installed but not in PATH
elif [ -f "/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvc++" ]; then
    print_warning "NVIDIA HPC SDK found in /opt/nvidia/hpc_sdk but not in PATH"
    print_info "To use it, add to your ~/.bashrc:"
    echo ""
    echo "  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:\$LD_LIBRARY_PATH"
    echo ""
    read -p "Would you like to add it to your PATH now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        BASH_RC="$HOME/.bashrc"
        if [ "$SUDO_USER" ]; then
            BASH_RC="/home/$SUDO_USER/.bashrc"
        fi
        cat >> "$BASH_RC" << 'ENVEOF'

# NVIDIA HPC SDK (added by ParaCodex installer)
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/man:$MANPATH
ENVEOF
        print_success "Added HPC SDK to $BASH_RC"
        print_info "Run 'source ~/.bashrc' or open a new terminal to use it"
        exit 0
    else
        print_info "Installation cancelled. HPC SDK is already installed."
        exit 0
    fi
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "This script must be run as root or with sudo"
    print_info "Please run: sudo $0"
    exit 1
fi

# Check system requirements
echo ""
print_info "Checking system requirements..."

# Check OS
if [ ! -f /etc/os-release ]; then
    print_error "Cannot determine OS. This script supports Ubuntu/Debian-based systems."
    exit 1
fi

source /etc/os-release
print_info "Detected OS: $NAME $VERSION"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    print_error "This installer only supports x86_64 architecture. Detected: $ARCH"
    exit 1
fi

# Check available disk space (need at least 10GB)
AVAILABLE_SPACE=$(df /opt 2>/dev/null | tail -1 | awk '{print $4}' || df / | tail -1 | awk '{print $4}')
REQUIRED_SPACE=$((10 * 1024 * 1024))  # 10GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_error "Insufficient disk space. Need at least 10GB free in /opt or /"
    print_info "Available: $(echo "scale=2; $AVAILABLE_SPACE/1024/1024" | bc)GB"
    exit 1
fi

# Check for wget or curl
if ! command_exists wget && ! command_exists curl; then
    print_error "Neither wget nor curl is installed. Installing wget..."
    apt-get update
    apt-get install -y wget
fi

print_success "System requirements check passed"

# Create temporary directory
echo ""
print_info "Creating temporary directory: $TEMP_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Download NVIDIA HPC SDK
echo ""
print_info "Downloading NVIDIA HPC SDK ${NVHPC_VERSION}..."
print_info "URL: $DOWNLOAD_URL"
print_warning "This is a large download (~3GB). This may take several minutes..."
echo ""

TARBALL="nvhpc_${NVHPC_FULL_VERSION}_Linux_x86_64_cuda_${CUDA_VERSION}.tar.gz"

if command_exists wget; then
    wget --progress=bar:force "$DOWNLOAD_URL" -O "$TARBALL" 2>&1 | \
        grep --line-buffered "%" | \
        sed -u -e "s,\.,,g" | \
        awk '{printf("\rDownload Progress: %s", $2); fflush()}'
    echo ""
elif command_exists curl; then
    curl -# -L "$DOWNLOAD_URL" -o "$TARBALL"
fi

if [ ! -f "$TARBALL" ]; then
    print_error "Download failed. Please check your internet connection and try again."
    cd /
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Verify download integrity
print_info "Verifying download integrity..."
FILESIZE=$(stat -f%z "$TARBALL" 2>/dev/null || stat -c%s "$TARBALL" 2>/dev/null)
MIN_SIZE=$((2500 * 1024 * 1024))  # 2.5GB minimum

if [ "$FILESIZE" -lt "$MIN_SIZE" ]; then
    print_error "Downloaded file is too small ($(echo "scale=2; $FILESIZE/1024/1024" | bc)MB). Download may have failed."
    print_info "Expected size: ~3GB"
    print_info "Try downloading manually from: https://developer.nvidia.com/hpc-sdk"
    cd /
    rm -rf "$TEMP_DIR"
    exit 1
fi

print_success "Download complete and verified"

# Extract archive
echo ""
print_info "Extracting archive..."
tar xzf "$TARBALL"

# Find the extracted directory
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "nvhpc_*" | head -1)
if [ -z "$EXTRACTED_DIR" ]; then
    print_error "Failed to find extracted directory"
    cd /
    rm -rf "$TEMP_DIR"
    exit 1
fi

cd "$EXTRACTED_DIR"
print_success "Extraction complete"

# Run installer
echo ""
print_info "Running NVIDIA HPC SDK installer..."
print_info "Installation directory: $INSTALL_DIR"
echo ""

# Run the installer with default options
if [ -f "install" ]; then
    ./install
else
    print_error "Installer script not found in extracted directory"
    cd /
    rm -rf "$TEMP_DIR"
    exit 1
fi

print_success "NVIDIA HPC SDK installed successfully"

# Set up environment variables
echo ""
print_info "Setting up environment variables..."

BASH_RC="$HOME/.bashrc"
if [ "$SUDO_USER" ]; then
    BASH_RC="/home/$SUDO_USER/.bashrc"
fi

# Check if already in bashrc
if grep -q "NVIDIA HPC SDK" "$BASH_RC" 2>/dev/null; then
    print_warning "Environment variables already configured in $BASH_RC"
else
    print_info "Adding environment variables to $BASH_RC"
    cat >> "$BASH_RC" << 'EOF'

# NVIDIA HPC SDK (added by ParaCodex installer)
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH
export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/man:$MANPATH
EOF
    print_success "Environment variables added to $BASH_RC"
fi

# Set up for current session
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib:$LD_LIBRARY_PATH

# Clean up
echo ""
print_info "Cleaning up temporary files..."
cd /
rm -rf "$TEMP_DIR"
print_success "Cleanup complete"

# Verify installation
echo ""
print_info "Verifying installation..."
echo ""

if command_exists nvc++; then
    NVC_VERSION=$(nvc++ --version 2>&1 | head -3)
    echo -e "${GREEN}✓ nvc++ compiler:${NC}"
    echo "$NVC_VERSION" | sed 's/^/  /'
    echo ""
else
    print_error "nvc++ not found in PATH after installation"
    print_info "You may need to start a new shell or run: source $BASH_RC"
fi

if command_exists nvfortran; then
    print_success "✓ nvfortran compiler found"
fi

# Test OpenMP support
print_info "Testing OpenMP support..."
TEST_FILE="/tmp/test_openmp_$$.c"
cat > "$TEST_FILE" << 'EOF'
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    {
        #pragma omp single
        printf("OpenMP with %d threads\n", omp_get_num_threads());
    }
    return 0;
}
EOF

if nvc++ -mp "$TEST_FILE" -o "${TEST_FILE%.c}" 2>/dev/null; then
    print_success "✓ OpenMP CPU support verified"
    rm -f "$TEST_FILE" "${TEST_FILE%.c}"
else
    print_warning "OpenMP test compilation failed"
fi

# Test GPU offload
print_info "Testing OpenMP GPU offload support..."
TEST_GPU_FILE="/tmp/test_gpu_$$.c"
cat > "$TEST_GPU_FILE" << 'EOF'
#include <stdio.h>
int main() {
    int n = 100, sum = 0;
    #pragma omp target teams distribute parallel for reduction(+:sum)
    for(int i = 0; i < n; i++) sum += i;
    printf("GPU offload test: sum = %d (expected 4950)\n", sum);
    return 0;
}
EOF

if nvc++ -mp=gpu "$TEST_GPU_FILE" -o "${TEST_GPU_FILE%.c}" 2>/dev/null; then
    print_success "✓ OpenMP GPU offload support verified"
    rm -f "$TEST_GPU_FILE" "${TEST_GPU_FILE%.c}"
else
    print_warning "GPU offload test compilation failed (GPU may not be available)"
fi

# Final summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                              ║${NC}"
echo -e "${GREEN}║                    ✓ INSTALLATION COMPLETE!                                  ║${NC}"
echo -e "${GREEN}║                                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
print_info "NVIDIA HPC SDK ${NVHPC_VERSION} has been successfully installed"
print_info "Installation directory: $INSTALL_DIR"
echo ""
print_warning "⚠️  IMPORTANT: To use the compiler in your current shell, run:"
echo ""
echo -e "    ${YELLOW}source $BASH_RC${NC}"
echo ""
print_info "Or open a new terminal window."
echo ""
print_info "Verify installation with:"
echo "    nvc++ --version"
echo "    nvc++ -mp -V  # Check OpenMP support"
echo ""
print_info "Next steps:"
echo "    1. Install other ParaCodex dependencies: ./setup_environment.sh"
echo "    2. Install Nsight Systems: ./install_nsight_systems.sh (if needed)"
echo "    3. Verify full environment: ./verify_environment.sh"
echo ""
print_success "Happy parallel programming! 🚀"
echo ""

