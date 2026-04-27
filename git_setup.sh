#!/bin/bash
# Set up git repo for llmgap, excluding large files and outputs

cd /cluster/home/tunguyen1/llmgap

# Create .gitignore if it doesn't exist or update it
cat > .gitignore << 'EOF'
# Generated files
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/

# Output and result files
out/
outputs/
*.csv
slurm-*.out
slurm-*.err

# Large external repos (use git submodules instead)
colm-paper-code-cleaned/
reasoning-efficiency/
reasoning-efficiency-v1/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.env
EOF

echo "✓ Created/updated .gitignore"

# Stage all important files
git add src/
git add inspect_results.py
git add evaluate_subquestions.py
git add test.py
git add *.sh
git add requirements.txt
git add qwen_gsm8k_inspection.jsonl
git add .gitignore
git add ICLR-2025-mathgap-out-of-distribution-evaluation-on-problems-with-arbitrarily-complex-proofs-Paper-Conference.pdf 2>/dev/null || true
git add srun.txt

echo "✓ Staged files for commit"

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short

# Commit
git commit -m "Add llmgap source code: finetune, inference, and evaluation scripts"

echo ""
echo "✓ Committed successfully"

# Show log
echo ""
echo "Recent commits:"
git log --oneline -3

echo ""
echo "Ready to push! Run:"
echo "  git push origin main"
echo "  (or replace 'main' with your default branch name)"
