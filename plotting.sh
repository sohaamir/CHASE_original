#!/bin/bash
#SBATCH --account=zhanglp-aamir-core
#SBATCH --qos=bbdefault
#SBATCH --ntasks=1
#SBATCH --job-name=llm_plotting
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=llm_plotting_%j.out
#SBATCH --error=llm_plotting_%j.err

########### plotting.sh ###########

echo "=================================================================================="
echo "LLM PLOTTING ONLY - Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Record start time
START_TIME=$(date +%s)

# Environment setup
export TERM=xterm-256color
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# ===== CONFIGURATION (MATCHING experiment.sh) =====
PROJECT_ROOT="/rds/projects/z/zhanglp-aamir-core/chase"
RESULTS_DIR="${PROJECT_ROOT}/results/llm_subset"  # ← CRITICAL: Match your output_dir
LOG_DIR="${PROJECT_ROOT}/logs"

# Create directories if needed
mkdir -p "${LOG_DIR}"

# Set working directory
cd "${PROJECT_ROOT}" || {
    echo "❌ ERROR: Could not change to project directory: ${PROJECT_ROOT}"
    exit 1
}
echo "✓ Working directory: $(pwd)"
echo ""

# ===== CLEANUP OLD LOGS =====
echo "🧹 Cleaning up old plotting logs..."
find . -maxdepth 1 -type f -name "llm_plotting_*" ! -name "*${SLURM_JOB_ID}*" -delete 2>/dev/null || true
echo "✓ Cleanup complete"
echo ""

# Exit on error
set -e

# ===== LOAD MODULES =====
echo "Loading modules..."
module purge
module load bluebear
module load bear-apps/2021a
module load MATLAB/2022a

# Verify MATLAB
if ! command -v matlab &> /dev/null; then
    echo "❌ ERROR: MATLAB not found in PATH"
    exit 1
fi

MATLAB_VERSION=$(matlab -batch "disp(version)" 2>/dev/null | tail -n 1)
echo "✓ MATLAB loaded: ${MATLAB_VERSION}"
echo ""

# ===== VERIFY PRE-COMPUTED RESULTS EXIST =====
echo "Verifying model fitting results..."

REQUIRED_FILES=(
    "${RESULTS_DIR}/fits_CHASE_table.mat"
    "${RESULTS_DIR}/model_comparison.mat"
)

ALL_FILES_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "${file}" ]]; then
        file_size=$(du -h "${file}" | cut -f1)
        echo "  ✓ ${file##*/} (${file_size})"
    else
        echo "  ❌ Missing: ${file}"
        ALL_FILES_PRESENT=false
    fi
done

if [[ "$ALL_FILES_PRESENT" = false ]]; then
    echo ""
    echo "❌ ERROR: Model fitting results not found in ${RESULTS_DIR}"
    echo "   Run model fitting first with: sbatch experiment.sh"
    exit 1
fi

echo ""
echo "✓ All required files found"
echo "  Results location: ${RESULTS_DIR}"
echo ""

# ===== CLEANUP FUNCTION =====
cleanup() {
    local exit_code=$?
    echo ""
    echo "=================================================================================="
    echo "🧹 Cleanup Process"
    echo "=================================================================================="
    
    # Kill any stray MATLAB processes
    pkill -u $USER matlab 2>/dev/null || true
    
    # Clean up java logs
    find . -maxdepth 1 -type f -name "java.log.*" -delete 2>/dev/null || true
    find /rds/homes/a/axs2210 -maxdepth 1 -type f -name "java.log.*" -delete 2>/dev/null || true
    
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo "Job Summary:"
    echo "  Exit code: ${exit_code}"
    echo "  Duration: ${MINUTES}m ${SECONDS}s"
    echo "  Ended: $(date)"
    
    if [[ $exit_code -eq 0 ]]; then
        echo "  Status: ✅ SUCCESS"
    else
        echo "  Status: ❌ FAILED"
    fi
    
    echo "=================================================================================="
    exit $exit_code
}

trap cleanup EXIT INT TERM

# ===== VERIFY PLOTTING SCRIPT EXISTS =====
PLOTTING_SCRIPT="BAKR_2024_results_and_figures.m"

if [[ ! -f "${PROJECT_ROOT}/${PLOTTING_SCRIPT}" ]]; then
    echo "❌ ERROR: Plotting script not found: ${PLOTTING_SCRIPT}"
    exit 1
fi
echo "✓ Plotting script found: ${PLOTTING_SCRIPT}"
echo ""

# ===== RUN PLOTTING =====
echo "=================================================================================="
echo "📈 GENERATING FIGURES"
echo "Script: ${PLOTTING_SCRIPT}"
echo "Started: $(date)"
echo "=================================================================================="
echo ""

# Run plotting script with unbuffered output and comprehensive logging
matlab -nodisplay -nosplash -nodesktop -r \
    "try; \
        fprintf('MATLAB started successfully\n'); \
        fprintf('Running ${PLOTTING_SCRIPT}...\n'); \
        BAKR_2024_results_and_figures; \
        fprintf('\n✓ Plotting completed successfully\n'); \
        exit(0); \
    catch ME; \
        fprintf('\n❌ ERROR in plotting:\n'); \
        fprintf('Message: %s\n', ME.message); \
        if ~isempty(ME.stack); \
            fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line); \
            fprintf('Full stack trace:\n'); \
            for i = 1:length(ME.stack); \
                fprintf('  [%d] %s (line %d)\n', i, ME.stack(i).name, ME.stack(i).line); \
            end; \
        end; \
        exit(1); \
    end" 2>&1 | tee llm_plotting_${SLURM_JOB_ID}.log

PLOT_EXIT=${PIPESTATUS[0]}

echo ""
echo "=================================================================================="
echo "Plotting Results:"
echo "  Exit code: ${PLOT_EXIT}"
echo "  Log: llm_plotting_${SLURM_JOB_ID}.log"

if [[ $PLOT_EXIT -ne 0 ]]; then
    echo "  Status: ❌ FAILED"
    echo "=================================================================================="
    echo ""
    echo "Plotting failed. Check log for details:"
    echo "  llm_plotting_${SLURM_JOB_ID}.log"
    exit $PLOT_EXIT
fi

echo "  Status: ✅ SUCCESS"
echo "=================================================================================="
echo ""

# ===== VERIFY FIGURE OUTPUTS =====
echo "Verifying generated figures..."

# Check for PNG files in results directory
n_png=$(find "${RESULTS_DIR}" -name "*.png" 2>/dev/null | wc -l)
n_fig=$(find "${RESULTS_DIR}" -name "*.fig" 2>/dev/null | wc -l)

echo "  Figures found:"
echo "    PNG files: ${n_png}"
echo "    FIG files: ${n_fig}"

if [[ $((n_png + n_fig)) -gt 0 ]]; then
    echo ""
    echo "  Generated files:"
    find "${RESULTS_DIR}" -type f \( -name "*.png" -o -name "*.fig" \) -exec basename {} \; | sort | sed 's/^/    - /'
fi

# ===== FINAL SUMMARY =====
echo ""
echo "=================================================================================="
echo "🎉 PLOTTING COMPLETE"
echo "=================================================================================="
echo ""
echo "Summary:"
echo "  Figures generated: $((n_png + n_fig))"
echo "  Duration: $((($(date +%s) - START_TIME) / 60))m $((($(date +%s) - START_TIME) % 60))s"
echo ""
echo "Outputs:"
echo "  Figures:  ${RESULTS_DIR}/*.png"
echo "  Logs:     llm_plotting_${SLURM_JOB_ID}.log"
echo "  SLURM:    llm_plotting_${SLURM_JOB_ID}.out/.err"
echo ""
echo "Job completed: $(date)"
echo "=================================================================================="

exit 0