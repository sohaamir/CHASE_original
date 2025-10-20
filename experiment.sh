#!/bin/bash
#SBATCH --account=zhanglp-aamir-core
#SBATCH --qos=bbdefault
#SBATCH --ntasks=1
#SBATCH --job-name=burgi_matlab_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=burgi_matlab_analysis_%j.out
#SBATCH --error=burgi_matlab_analysis_%j.err

########### experiment.sh ###########

echo "=================================================================================="
echo "LLM MATLAB ANALYSIS - Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Record start time
START_TIME=$(date +%s)

# Environment setup
export TERM=xterm-256color
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# ===== CONFIGURATION =====
PROJECT_ROOT="/rds/projects/z/zhanglp-aamir-core/chase"
RESULTS_DIR="${PROJECT_ROOT}/results/burgi_analysis"
LOG_DIR="${PROJECT_ROOT}/logs"

# Create directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Set working directory
cd "${PROJECT_ROOT}" || {
    echo "❌ ERROR: Could not change to project directory: ${PROJECT_ROOT}"
    exit 1
}
echo "✓ Working directory: $(pwd)"

# ===== COMPREHENSIVE CLEANUP OF OLD FILES =====
echo ""
echo "=================================================================================="
echo "🧹 Cleaning up old temporary files and logs..."
echo "=================================================================================="

# Count files before cleanup in main directory
OLD_COUNT=$(find . -maxdepth 1 -type f \( \
    -name "matlab_analysis_*" \
    \) ! -name "*${SLURM_JOB_ID}*" 2>/dev/null | wc -l)

echo "Found ${OLD_COUNT} old matlab_analysis_* files in main directory"

# Remove ALL old matlab_analysis files (including .out, .err, .log extensions)
echo "Removing ALL old matlab_analysis_* files from main directory..."
find . -maxdepth 1 -type f -name "matlab_analysis_*" ! -name "*${SLURM_JOB_ID}*" -delete 2>/dev/null || true

# Verify cleanup
REMAINING_COUNT=$(find . -maxdepth 1 -type f -name "matlab_analysis_*" ! -name "*${SLURM_JOB_ID}*" 2>/dev/null | wc -l)
echo "  Removed: $((OLD_COUNT - REMAINING_COUNT)) files"
echo "  Remaining (should be 0): ${REMAINING_COUNT}"

# Remove old log files and matlab_analysis files from logs directory
if [[ -d "${LOG_DIR}" ]]; then
    echo ""
    echo "Cleaning old files from ${LOG_DIR}..."
    
    # Count files before cleanup
    OLD_LOG_COUNT=$(find "${LOG_DIR}" -type f \( -name "*.log" -o -name "matlab_analysis_*" \) 2>/dev/null | wc -l)
    echo "  Found ${OLD_LOG_COUNT} files before cleanup"
    
    # Remove old log files AND matlab_analysis files from logs directory
    find "${LOG_DIR}" -type f \( -name "*.log" -o -name "matlab_analysis_*" \) ! -name "*${SLURM_JOB_ID}*" -delete 2>/dev/null || true
    
    # Count remaining
    NEW_LOG_COUNT=$(find "${LOG_DIR}" -type f \( -name "*.log" -o -name "matlab_analysis_*" \) 2>/dev/null | wc -l)
    echo "  Removed: $((OLD_LOG_COUNT - NEW_LOG_COUNT)) files"
    echo "  Remaining: ${NEW_LOG_COUNT}"
fi

# Remove MATLAB temporary files from both project and home directory
echo ""
echo "Removing MATLAB temporary files..."

# In project directory
JAVA_LOGS_PROJECT=$(find . -maxdepth 1 -type f -name "java.log.*" 2>/dev/null | wc -l)
if [[ $JAVA_LOGS_PROJECT -gt 0 ]]; then
    echo "  Removing ${JAVA_LOGS_PROJECT} java.log files from project directory"
    find . -maxdepth 1 -type f -name "java.log.*" -delete 2>/dev/null || true
fi

find . -maxdepth 1 -type f -name "matlab_crash_dump.*" -delete 2>/dev/null || true

# In home directory
JAVA_LOGS_HOME=$(find /rds/homes/a/axs2210 -maxdepth 1 -type f -name "java.log.*" 2>/dev/null | wc -l)
if [[ $JAVA_LOGS_HOME -gt 0 ]]; then
    echo "  Removing ${JAVA_LOGS_HOME} java.log files from home directory"
    find /rds/homes/a/axs2210 -maxdepth 1 -type f -name "java.log.*" -delete 2>/dev/null || true
fi

# Clean up empty directories in logs
if [[ -d "${LOG_DIR}" ]]; then
    find "${LOG_DIR}" -type d -empty -delete 2>/dev/null || true
fi

echo ""
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
    echo "Available modules:"
    module avail MATLAB
    exit 1
fi

MATLAB_VERSION=$(matlab -batch "disp(version)" 2>/dev/null | tail -n 1)
echo "✓ MATLAB loaded: ${MATLAB_VERSION}"

# ===== SYSTEM INFO =====
echo ""
echo "System Information:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
echo "  Time limit: ${SLURM_TIMELIMIT}"
echo ""

# ===== MEMORY MONITORING =====
monitor_memory() {
    local log_file="${LOG_DIR}/memory_usage_${SLURM_JOB_ID}.log"
    echo "Time,MemoryUsedGB,MemoryTotalGB" > "${log_file}"
    
    while true; do
        memory_used=$(free -g | awk '/^Mem:/ {print $3}')
        memory_total=$(free -g | awk '/^Mem:/ {print $2}')
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "${timestamp},${memory_used},${memory_total}" >> "${log_file}"
        echo "[$(date '+%H:%M:%S')] Memory: ${memory_used}GB/${memory_total}GB"
        sleep 300  # Log every 5 minutes
    done
}

monitor_memory &
MONITOR_PID=$!
echo "✓ Memory monitoring started (PID: ${MONITOR_PID})"
echo "  Log: ${LOG_DIR}/memory_usage_${SLURM_JOB_ID}.log"
echo ""

# ===== CLEANUP FUNCTION =====
cleanup() {
    local exit_code=$?
    echo ""
    echo "=================================================================================="
    echo "🧹 Cleanup Process"
    echo "=================================================================================="
    
    # Stop memory monitor
    if kill -0 $MONITOR_PID 2>/dev/null; then
        kill $MONITOR_PID 2>/dev/null || true
        echo "✓ Stopped memory monitor"
    fi
    
    # Kill any stray MATLAB processes
    pkill -u $USER matlab 2>/dev/null || true
    
    # Clean up java logs created during this run
    find . -maxdepth 1 -type f -name "java.log.*" -delete 2>/dev/null || true
    find /rds/homes/a/axs2210 -maxdepth 1 -type f -name "java.log.*" -newer matlab_analysis_${SLURM_JOB_ID}.out -delete 2>/dev/null || true
    
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo "Job Summary:"
    echo "  Exit code: ${exit_code}"
    echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "  Ended: $(date)"
    
    if [[ $exit_code -eq 0 ]]; then
        echo "  Status: ✅ SUCCESS"
    elif [[ $exit_code -eq 124 ]]; then
        echo "  Status: ⏰ TIMEOUT"
    else
        echo "  Status: ❌ FAILED"
    fi
    
    echo "=================================================================================="
    exit $exit_code
}

trap cleanup EXIT INT TERM

# ===== SCRIPT VERIFICATION =====
echo "Verifying scripts..."

# Check for required scripts
REQUIRED_SCRIPTS=(
    "BAKR_2024_run_model_fitting.m"
    "BAKR_2024_results_and_figures.m"
)
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -f "${PROJECT_ROOT}/${script}" ]]; then
        echo "❌ ERROR: Required script not found: ${script}"
        exit 1
    fi
done
echo "✓ All required scripts found"
echo ""

# ===== STAGE 1: MODEL FITTING =====
echo "=================================================================================="
echo "📊 STAGE 1: MODEL FITTING"
echo "Script: BAKR_2024_run_model_fitting.m"
echo "Started: $(date)"
echo "=================================================================================="
echo ""

FITTING_START=$(date +%s)

# Run model fitting script with unbuffered output
matlab -nodisplay -nosplash -nodesktop -r \
    "try; \
        fprintf('MATLAB started successfully\n'); \
        fprintf('Running BAKR_2024_run_model_fitting.m...\n'); \
        BAKR_2024_run_model_fitting; \
        fprintf('\n✓ Model fitting completed successfully\n'); \
        exit(0); \
    catch ME; \
        fprintf('\n❌ ERROR in model fitting:\n'); \
        fprintf('Message: %s\n', ME.message); \
        fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line); \
        exit(1); \
    end" 2>&1 | stdbuf -o0 tee -a matlab_analysis_${SLURM_JOB_ID}.log

FITTING_EXIT_CODE=${PIPESTATUS[0]}
FITTING_END=$(date +%s)
FITTING_DURATION=$((FITTING_END - FITTING_START))

echo ""
echo "=================================================================================="
echo "Model Fitting Results:"
echo "  Exit code: ${FITTING_EXIT_CODE}"
echo "  Duration: $((FITTING_DURATION / 60)) minutes"
echo "  Log: matlab_analysis_${SLURM_JOB_ID}.log"

if [[ $FITTING_EXIT_CODE -ne 0 ]]; then
    echo "  Status: ❌ FAILED"
    echo "=================================================================================="
    echo ""
    echo "Model fitting failed. Check log for details:"
    echo "  matlab_analysis_${SLURM_JOB_ID}.log"
    exit $FITTING_EXIT_CODE
fi

echo "  Status: ✅ SUCCESS"
echo "=================================================================================="
echo ""

# Verify outputs were created
EXPECTED_OUTPUTS=(
    "${RESULTS_DIR}/fits_CHASE_struct.mat"
    "${RESULTS_DIR}/fits_CHASE_table.mat"
    "${RESULTS_DIR}/model_comparison.mat"
)

echo "Verifying model fitting outputs..."
ALL_OUTPUTS_PRESENT=true
for output in "${EXPECTED_OUTPUTS[@]}"; do
    if [[ -f "${output}" ]]; then
        file_size=$(du -h "${output}" | cut -f1)
        echo "  ✓ ${output##*/} (${file_size})"
    else
        echo "  ❌ Missing: ${output##*/}"
        ALL_OUTPUTS_PRESENT=false
    fi
done

if [[ "$ALL_OUTPUTS_PRESENT" = false ]]; then
    echo ""
    echo "⚠️  WARNING: Some expected outputs are missing"
    echo "   Proceeding to plotting stage anyway..."
fi
echo ""

sleep 2  # Brief pause between stages

# ===== STAGE 2: RESULTS AND FIGURES =====
echo "=================================================================================="
echo "📈 STAGE 2: RESULTS AND FIGURES"
echo "Script: BAKR_2024_results_and_figures.m"
echo "Started: $(date)"
echo "=================================================================================="
echo ""

PLOTTING_START=$(date +%s)

# Run results and figures script with unbuffered output
matlab -nodisplay -nosplash -nodesktop -r \
    "try; \
        fprintf('MATLAB started successfully\n'); \
        fprintf('Running BAKR_2024_results_and_figures.m...\n'); \
        BAKR_2024_results_and_figures; \
        fprintf('\n✓ Results and figures completed successfully\n'); \
        exit(0); \
    catch ME; \
        fprintf('\n❌ ERROR in results/figures:\n'); \
        fprintf('Message: %s\n', ME.message); \
        fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line); \
        exit(1); \
    end" 2>&1 | stdbuf -o0 tee -a matlab_analysis_${SLURM_JOB_ID}.log

PLOTTING_EXIT_CODE=${PIPESTATUS[0]}
PLOTTING_END=$(date +%s)
PLOTTING_DURATION=$((PLOTTING_END - PLOTTING_START))

echo ""
echo "=================================================================================="
echo "Results & Figures:"
echo "  Exit code: ${PLOTTING_EXIT_CODE}"
echo "  Duration: $((PLOTTING_DURATION / 60)) minutes"
echo "  Log: matlab_analysis_${SLURM_JOB_ID}.log"

if [[ $PLOTTING_EXIT_CODE -ne 0 ]]; then
    echo "  Status: ❌ FAILED"
    echo "=================================================================================="
    echo ""
    echo "Results/figures generation failed. Check log for details:"
    echo "  matlab_analysis_${SLURM_JOB_ID}.log"
    exit $PLOTTING_EXIT_CODE
fi

echo "  Status: ✅ SUCCESS"
echo "=================================================================================="
echo ""

# ===== VERIFY OUTPUTS =====
echo "Verifying figure outputs..."
PLOTS_DIR="${RESULTS_DIR}/plots"
if [[ -d "${PLOTS_DIR}" ]]; then
    n_figures=$(find "${PLOTS_DIR}" -name "*.png" -o -name "*.fig" | wc -l)
    plots_size=$(du -sh "${PLOTS_DIR}" | cut -f1)
    echo "  ✓ Plots directory: ${PLOTS_DIR}"
    echo "  ✓ Figures generated: ${n_figures}"
    echo "  ✓ Total size: ${plots_size}"
else
    echo "  ⚠️  Plots directory not found: ${PLOTS_DIR}"
fi

# ===== FINAL SUMMARY =====
echo ""
echo "=================================================================================="
echo "🎉 ANALYSIS COMPLETE - ALL STAGES SUCCESSFUL"
echo "=================================================================================="
echo ""
echo "Summary:"
echo "  Model Fitting:    ✅ Success ($((FITTING_DURATION / 60))min)"
echo "  Results/Figures:  ✅ Success ($((PLOTTING_DURATION / 60))min)"
echo "  Total Duration:   $((($(date +%s) - START_TIME) / 60)) minutes"
echo ""
echo "Outputs:"
echo "  Results:  ${RESULTS_DIR}"
echo "  Figures:  ${PLOTS_DIR}"
echo "  Logs:     ${LOG_DIR}"
echo ""

# List all generated figures
if [[ -d "${PLOTS_DIR}" ]]; then
    echo "Generated Figures:"
    find "${PLOTS_DIR}" -type f \( -name "*.png" -o -name "*.fig" \) -exec basename {} \; | sort | sed 's/^/  - /'
    echo ""
fi

echo "Logs saved to:"
echo "  - matlab_analysis_${SLURM_JOB_ID}.log (master log)"
echo "  - matlab_analysis_${SLURM_JOB_ID}.out (SLURM output)"
echo "  - matlab_analysis_${SLURM_JOB_ID}.err (SLURM errors)"
echo "  - ${LOG_DIR}/memory_usage_${SLURM_JOB_ID}.log (memory monitoring)"
echo ""
echo "Job completed: $(date)"
echo "=================================================================================="

exit 0