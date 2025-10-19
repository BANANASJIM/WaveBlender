#!/bin/bash
# WaveBlender One-Click Setup Script
# This script handles: dependencies check, building, Python environment, and scene downloads

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "==================================="
echo "WaveBlender One-Click Setup"
echo "==================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Check dependencies
echo "Step 1: Checking system dependencies..."
echo "---------------------------------------"

MISSING_DEPS=()

if ! command_exists cmake; then
    MISSING_DEPS+=("cmake")
fi

if ! command_exists nvcc; then
    MISSING_DEPS+=("cuda")
fi

if ! command_exists python; then
    MISSING_DEPS+=("python")
fi

if ! command_exists git; then
    MISSING_DEPS+=("git")
fi

# Check for Eigen
if [ ! -f "/usr/include/eigen3/Eigen/Core" ] && [ ! -f "/usr/local/include/eigen3/Eigen/Core" ]; then
    MISSING_DEPS+=("eigen")
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    print_error "Missing dependencies: ${MISSING_DEPS[*]}"
    echo ""
    echo "Please install the missing dependencies:"
    echo "  Arch Linux: sudo pacman -S cmake cuda eigen python git"
    echo "  Ubuntu/Debian: sudo apt install cmake eigen3 python3 python3-venv git"
    echo "  CUDA: Download from https://developer.nvidia.com/cuda-downloads"
    exit 1
else
    print_status "All system dependencies found"
fi

echo ""

# Step 2: Initialize git submodules
echo "Step 2: Initializing git submodules..."
echo "---------------------------------------"
if [ -d ".git" ]; then
    git submodule update --init --recursive
    print_status "Git submodules initialized"
else
    print_warning "Not a git repository, skipping submodule init"
fi

echo ""

# Step 3: Setup Python virtual environment
echo "Step 3: Setting up Python virtual environment..."
echo "---------------------------------------"
if [ ! -d "venv" ]; then
    python -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q numpy matplotlib scipy soundfile

print_status "Python dependencies installed"
echo ""

# Step 4: Build WaveBlender
echo "Step 4: Building WaveBlender..."
echo "---------------------------------------"

mkdir -p build
cd build

print_status "Running CMake configuration..."
cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null

print_status "Compiling (this may take a few minutes)..."
make -j$(nproc) 2>&1 | grep -E "Built target|error|warning" || true

if [ -f "WaveBlender" ]; then
    print_status "WaveBlender built successfully!"
    ls -lh WaveBlender
else
    print_error "Build failed!"
    exit 1
fi

cd ..
echo ""

# Step 5: Download additional scenes (optional)
echo "Step 5: Download additional scenes (optional)"
echo "---------------------------------------"
echo "Available scenes to download:"
echo "  1. Glass Pour (75.4MB)"
echo "  2. Paddle Splash (63.3MB)"
echo "  3. Lego Drop (148MB)"
echo "  4. Spolling Bowl (135MB)"
echo "  5. All scenes (total ~422MB)"
echo "  s. Skip download"
echo ""

read -p "Enter your choice (1-5, s to skip): " scene_choice

if [ "$scene_choice" != "s" ] && [ "$scene_choice" != "S" ]; then
    SCENES_DIR="Scenes"
    BASE_URL="https://graphics.stanford.edu/papers/waveblender/dataset/data"

    mkdir -p "$SCENES_DIR"
    cd "$SCENES_DIR"

    download_scene() {
        local scene_name=$1
        local zip_file=$2
        local size=$3

        if [ -d "$scene_name" ]; then
            print_warning "$scene_name already exists, skipping..."
            return
        fi

        echo "Downloading $scene_name ($size)..."
        wget -q --show-progress "$BASE_URL/$zip_file" -O "$zip_file"
        unzip -q "$zip_file"
        rm "$zip_file"
        print_status "$scene_name downloaded and extracted"
    }

    case $scene_choice in
        1) download_scene "GlassPour" "GlassPour.zip" "75.4MB" ;;
        2) download_scene "PaddleSplash" "PaddleSplash.zip" "63.3MB" ;;
        3) download_scene "LegoDrop" "LegoDrop.zip" "148MB" ;;
        4) download_scene "SpollingBowl" "SpollingBowl.zip" "135MB" ;;
        5)
            download_scene "GlassPour" "GlassPour.zip" "75.4MB"
            download_scene "PaddleSplash" "PaddleSplash.zip" "63.3MB"
            download_scene "LegoDrop" "LegoDrop.zip" "148MB"
            download_scene "SpollingBowl" "SpollingBowl.zip" "135MB"
            ;;
        *)
            print_warning "Invalid choice, skipping scene download"
            ;;
    esac

    cd ..
else
    print_status "Skipping additional scene downloads"
fi

echo ""
echo "==================================="
echo "✅ Setup Complete!"
echo "==================================="
echo ""
echo "To test the installation, run:"
echo "  cd build"
echo "  ./WaveBlender ../Scenes/CupPhone/config.json"
echo "  source ../venv/bin/activate"
echo "  python ../scripts/write_wav.py CupPhone_out.bin 88200"
echo ""
echo "Available scenes:"
ls -1 Scenes/ 2>/dev/null || echo "  CupPhone (included by default)"
echo ""
echo "To activate Python environment in the future:"
echo "  source venv/bin/activate"
echo ""
