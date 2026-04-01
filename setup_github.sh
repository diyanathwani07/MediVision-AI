#!/bin/bash
# =============================================================
# MediVision AI — GitHub Setup Script
# Run this once after downloading the project folder
# =============================================================

echo "🚀 MediVision AI — GitHub Setup"
echo "================================"

# 1. Initialize git
git init
echo "✅ Git initialized"

# 2. Add all files
git add .
git commit -m "🧠 Initial commit: MediVision AI - Brain Tumor Classification Platform

Features:
- Custom CNN from scratch with Keras Tuner hyperparameter search
- VGG16 Transfer Learning (pretrained vs from-scratch comparison)
- DCGAN for synthetic medical image generation/augmentation  
- Flask real-time web dashboard with Grad-CAM visualization
- Ensemble CNN + VGG16 inference
- MLflow experiment tracking
- Full test suite with pytest
- GitHub Actions CI/CD pipeline"

echo "✅ Initial commit created"

# 3. Prompt for GitHub URL
echo ""
echo "Now go to https://github.com/new and create a repo called 'MediVision-AI'"
echo "Then paste your repo URL below:"
read -p "GitHub repo URL (e.g. https://github.com/yourusername/MediVision-AI.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "⚠️  No URL entered. Run manually:"
    echo "    git remote add origin YOUR_REPO_URL"
    echo "    git branch -M main"
    echo "    git push -u origin main"
else
    git remote add origin "$REPO_URL"
    git branch -M main
    git push -u origin main
    echo "✅ Pushed to GitHub!"
    echo "🎉 Your project is live at: ${REPO_URL%.git}"
fi
