#!/bin/bash
#
# Quick Deployment Script for CLaRa Project
#

set -e

echo "========================================="
echo "  CLaRa Project Deployment Script"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Show current status
echo -e "${YELLOW}Step 1: Checking current git status...${NC}"
git status --short
echo ""

# Step 2: Ask for confirmation
echo -e "${YELLOW}Step 2: Do you want to commit all changes?${NC}"
echo "This will add and commit all modified and new files."
read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Deployment cancelled.${NC}"
    exit 1
fi

# Step 3: Add all files
echo -e "${YELLOW}Step 3: Adding all files...${NC}"
git add .

# Step 4: Show what will be committed
echo -e "${YELLOW}Step 4: Files to be committed:${NC}"
git status --short
echo ""

# Step 5: Commit
echo -e "${YELLOW}Step 5: Creating commit...${NC}"
git commit -F /tmp/commit_message.txt || {
    echo -e "${RED}Commit failed. Please check for errors.${NC}"
    exit 1
}
echo -e "${GREEN}✓ Commit created successfully!${NC}"
echo ""

# Step 6: Ask about remote repository
echo -e "${YELLOW}Step 6: Configure remote repository${NC}"
echo ""
echo "Choose an option:"
echo "  1) Push to existing origin (keep current remote)"
echo "  2) Create new private repository (will guide you)"
echo "  3) Skip pushing (commit only)"
read -p "Enter choice (1/2/3): " -n 1 -r choice
echo ""

case $choice in
    1)
        echo -e "${YELLOW}Pushing to existing origin...${NC}"
        git push origin main || git push origin master
        echo -e "${GREEN}✓ Pushed successfully!${NC}"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}To create a new private repository:${NC}"
        echo ""
        echo "1. Visit: https://github.com/new"
        echo "2. Repository name: ml-clara-private (or your choice)"
        echo "3. Select: ✅ Private"
        echo "4. Do NOT initialize with README, .gitignore, or license"
        echo "5. Click 'Create repository'"
        echo ""
        read -p "Press Enter when repository is created..."
        echo ""
        read -p "Enter your GitHub username: " username
        read -p "Enter repository name (default: ml-clara-private): " reponame
        reponame=${reponame:-ml-clara-private}

        echo ""
        echo -e "${YELLOW}Updating remote repository...${NC}"

        # Check if origin exists
        if git remote | grep -q "^origin$"; then
            echo "Removing old origin..."
            git remote remove origin
        fi

        # Add new origin
        echo "Adding new remote: https://github.com/$username/$reponame.git"
        git remote add origin "https://github.com/$username/$reponame.git"

        # Push
        echo -e "${YELLOW}Pushing to new repository...${NC}"
        git branch -M main
        git push -u origin main

        echo ""
        echo -e "${GREEN}✓ Successfully pushed to new private repository!${NC}"
        echo -e "${GREEN}✓ Repository URL: https://github.com/$username/$reponame${NC}"
        ;;
    3)
        echo -e "${YELLOW}Skipping push. Your changes are committed locally.${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Skipping push.${NC}"
        ;;
esac

echo ""
echo "========================================="
echo -e "${GREEN}  Deployment Complete! ${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify your repository at: https://github.com/YOUR_USERNAME/YOUR_REPO"
echo "2. Check that it's marked as 🔒 Private"
echo "3. Review DEPLOY_GUIDE.md for detailed deployment options"
echo ""
echo "For team collaboration, see DEPLOY_GUIDE.md section 'Team Collaboration'"
echo ""
