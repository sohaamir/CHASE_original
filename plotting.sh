#!/bin/bash
#SBATCH --account=zhanglp-aamir-core
#SBATCH --qos=bbdefault
#SBATCH --ntasks=1
#SBATCH --job-name=llm_plotting
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=matlab_plotting_%j.out
#SBATCH --error=matlab_plotting_%j.err

########### plotting.sh ###########

echo "=================================================================================="
echo "LLM PLOTTING ONLY - Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=================================================================================="

# Environment setup
export TERM=xterm-256color
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Configuration
PROJECT_ROOT="/rds/projects/z/zhanglp-aamir-core/osf"
RESULTS_DIR="${PROJECT_ROOT}/results/llm_analysis"

# Set working directory
cd "${PROJECT_ROOT}" || {
    echo "❌ ERROR: Could not change to project directory"
    exit 1
}

echo "✓ Working directory: $(pwd)"

# Exit on error
set -e

# Load MATLAB
echo "Loading modules..."
module purge
module load bluebear
module load bear-apps/2021a
module load MATLAB/2022a

echo "✓ MATLAB loaded"

# Verify results exist
if [[ ! -f "${RESULTS_DIR}/fits_CHASE_struct.mat" ]]; then
    echo "❌ ERROR: Model fitting results not found!"
    echo "   Run model fitting first with: sbatch experiment.sh"
    exit 1
fi

echo "✓ Found pre-computed results in: ${RESULTS_DIR}"
echo ""

# Run plotting only
echo "=================================================================================="
echo "📈 GENERATING FIGURES"
echo "=================================================================================="

matlab -nodisplay -nosplash -nodesktop -r \
    "try; \
        fprintf('Running AS_2025_results_and_figures.m...\n'); \
        AS_2025_results_and_figures; \
        fprintf('\n✓ Plotting completed successfully\n'); \
        exit(0); \
    catch ME; \
        fprintf('\n❌ ERROR in plotting:\n'); \
        fprintf('Message: %s\n', ME.message); \
        fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line); \
        exit(1); \
    end"

PLOT_EXIT=$?

echo ""
echo "=================================================================================="
if [[ $PLOT_EXIT -eq 0 ]]; then
    echo "✅ PLOTTING COMPLETE"
    echo "Figures saved to: ${RESULTS_DIR}/plots"
    ls -lh "${RESULTS_DIR}/plots/" 2>/dev/null || echo "No plots directory found"
else
    echo "❌ PLOTTING FAILED"
fi
echo "=================================================================================="

exit $PLOT_EXIT