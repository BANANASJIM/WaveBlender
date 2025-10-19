#!/bin/bash
# WaveBlender Quick Run Script

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Check if WaveBlender is built
if [ ! -f "build/WaveBlender" ]; then
    echo "❌ WaveBlender not built! Please run ./setup.sh first"
    exit 1
fi

# Activate Python environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Warning: Python virtual environment not found"
fi

# List available scenes
echo "Available scenes:"
echo "================="
scene_num=1
declare -a scenes
for scene_dir in Scenes/*/; do
    if [ -d "$scene_dir" ]; then
        scene_name=$(basename "$scene_dir")
        if [ -f "${scene_dir}config.json" ]; then
            echo "  $scene_num. $scene_name"
            scenes[$scene_num]="$scene_name"
            ((scene_num++))
        fi
    fi
done

if [ ${#scenes[@]} -eq 0 ]; then
    echo "❌ No scenes found in Scenes/ directory"
    exit 1
fi

echo ""
read -p "Select scene number to run (1-$((scene_num-1))): " choice

if [ -z "${scenes[$choice]}" ]; then
    echo "❌ Invalid choice"
    exit 1
fi

selected_scene="${scenes[$choice]}"
config_file="Scenes/${selected_scene}/config.json"
output_name="${selected_scene}_out"

echo ""
echo "🎬 Running WaveBlender with $selected_scene..."
echo "================================================"

cd build

# Run simulation
./WaveBlender "../$config_file"

if [ $? -ne 0 ]; then
    echo "❌ Simulation failed"
    exit 1
fi

echo ""
echo "🔊 Generating WAV file..."

# Check for output .bin file
if [ ! -f "${output_name}.bin" ]; then
    echo "⚠️  Output file not found, searching..."
    # Try to find any .bin file that was just created
    output_bin=$(ls -t *.bin 2>/dev/null | head -1)
    if [ -z "$output_bin" ]; then
        echo "❌ No output .bin file found"
        exit 1
    fi
    output_name="${output_bin%.bin}"
fi

# Detect sample rate from config or use default
sample_rate=88200
if [ -f "../$config_file" ]; then
    # Try to extract sample rate from config
    detected_rate=$(grep -o '"sampleRate"[[:space:]]*:[[:space:]]*[0-9]*' "../$config_file" | grep -o '[0-9]*$')
    if [ ! -z "$detected_rate" ]; then
        sample_rate=$detected_rate
    fi
fi

echo "Sample rate: $sample_rate Hz"
python ../scripts/write_wav.py "${output_name}.bin" $sample_rate

if [ -f "${output_name}.wav" ]; then
    echo ""
    echo "✅ Success!"
    echo "================================================"
    echo "Output file: build/${output_name}.wav"
    ls -lh "${output_name}.wav"
    echo ""
    echo "To play the audio:"
    echo "  mpv build/${output_name}.wav"
    echo "  # or"
    echo "  ffplay build/${output_name}.wav"
else
    echo "❌ WAV generation failed"
    exit 1
fi
