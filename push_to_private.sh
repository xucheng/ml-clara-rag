#!/bin/bash
#
# Quick script to push to your private repository
# Supports both SSH and HTTPS authentication methods
#
# Usage:
#   ./push_to_private.sh USERNAME [REPO_NAME]
#
# Examples:
#   ./push_to_private.sh xucheng
#   ./push_to_private.sh xucheng ml-clara-rag
#

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if username provided
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: GitHub username required${NC}"
    echo ""
    echo "Usage: ./push_to_private.sh USERNAME [REPO_NAME]"
    echo ""
    echo "Examples:"
    echo "  ./push_to_private.sh xucheng"
    echo "  ./push_to_private.sh xucheng ml-clara-rag"
    echo ""
    exit 1
fi

USERNAME=$1
REPO_NAME=${2:-ml-clara-private}

echo "========================================="
echo "  Pushing to Private Repository"
echo "========================================="
echo ""
echo -e "${BLUE}GitHub Username:${NC} $USERNAME"
echo -e "${BLUE}Repository Name:${NC} $REPO_NAME"
echo ""

# Check if SSH key is configured
SSH_CONFIGURED=false
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    SSH_CONFIGURED=true
elif [ -f ~/.ssh/id_rsa ] || [ -f ~/.ssh/id_ed25519 ] || [ -f ~/.ssh/id_ecdsa ]; then
    SSH_CONFIGURED=true
fi

# Show authentication method options
echo -e "${YELLOW}Choose authentication method:${NC}"
echo ""

if [ "$SSH_CONFIGURED" = true ]; then
    echo "  1) SSH (Recommended - No password needed)"
    echo "     URL: git@github.com:$USERNAME/$REPO_NAME.git"
    echo ""
    echo "  2) HTTPS (Requires Personal Access Token)"
    echo "     URL: https://github.com/$USERNAME/$REPO_NAME.git"
    echo ""
    DEFAULT_CHOICE="1"
    echo -e "${GREEN}✓ SSH key detected - Option 1 recommended${NC}"
else
    echo "  1) SSH (Requires SSH key setup)"
    echo "     URL: git@github.com:$USERNAME/$REPO_NAME.git"
    echo ""
    echo "  2) HTTPS (Requires Personal Access Token)"
    echo "     URL: https://github.com/$USERNAME/$REPO_NAME.git"
    echo ""
    DEFAULT_CHOICE="2"
    echo -e "${YELLOW}⚠️  No SSH key detected - Option 2 recommended${NC}"
fi

echo ""
read -p "Enter choice (1 or 2) [default: $DEFAULT_CHOICE]: " auth_choice
auth_choice=${auth_choice:-$DEFAULT_CHOICE}

# Set remote URL based on choice
if [ "$auth_choice" = "1" ]; then
    REMOTE_URL="git@github.com:$USERNAME/$REPO_NAME.git"
    AUTH_METHOD="SSH"

    # Check SSH connection
    echo ""
    echo "Testing SSH connection to GitHub..."
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo -e "${GREEN}✓ SSH authentication successful${NC}"
    else
        echo -e "${YELLOW}⚠️  SSH connection test inconclusive${NC}"
        echo ""
        echo "If push fails, you may need to set up SSH key:"
        echo "1. Generate key: ssh-keygen -t ed25519 -C \"your_email@example.com\""
        echo "2. Add to GitHub: https://github.com/settings/keys"
        echo ""
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 1
        fi
    fi
elif [ "$auth_choice" = "2" ]; then
    REMOTE_URL="https://github.com/$USERNAME/$REPO_NAME.git"
    AUTH_METHOD="HTTPS"
else
    echo -e "${RED}Invalid choice. Exiting.${NC}"
    exit 1
fi

# Summary
echo ""
echo "========================================="
echo -e "${BLUE}Configuration Summary${NC}"
echo "========================================="
echo -e "Username:     ${GREEN}$USERNAME${NC}"
echo -e "Repository:   ${GREEN}$REPO_NAME${NC}"
echo -e "Auth Method:  ${GREEN}$AUTH_METHOD${NC}"
echo -e "Remote URL:   ${GREEN}$REMOTE_URL${NC}"
echo "========================================="
echo ""

# Confirm
read -p "Proceed with these settings? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Remove old origin
echo ""
echo "Removing old origin..."
if git remote remove origin 2>/dev/null; then
    echo -e "${GREEN}✓ Old origin removed${NC}"
else
    echo "No origin to remove"
fi

# Add new origin
echo ""
echo "Adding new origin..."
git remote add origin "$REMOTE_URL"
echo -e "${GREEN}✓ New origin added${NC}"
echo ""
git remote -v

# Show credential information for HTTPS
if [ "$AUTH_METHOD" = "HTTPS" ]; then
    echo ""
    echo "========================================="
    echo -e "${YELLOW}⚠️  Authentication Required${NC}"
    echo "========================================="
    echo ""
    echo "You will be prompted for credentials:"
    echo "  Username: $USERNAME"
    echo "  Password: YOUR_PERSONAL_ACCESS_TOKEN"
    echo ""
    echo -e "${RED}⚠️  Use Personal Access Token, NOT your GitHub password!${NC}"
    echo ""
    echo "Get your token at:"
    echo "  https://github.com/settings/tokens"
    echo ""
    echo "Token setup:"
    echo "  1. Click 'Generate new token (classic)'"
    echo "  2. Select 'repo' scope"
    echo "  3. Copy the generated token"
    echo ""
    read -p "Press Enter when ready to push..."
fi

# Push
echo ""
echo "Pushing to private repository..."
git branch -M main

if git push -u origin main; then
    echo ""
    echo "========================================="
    echo -e "${GREEN}  ✅ Successfully Pushed!${NC}"
    echo "========================================="
    echo ""
    echo -e "Repository URL: ${BLUE}https://github.com/$USERNAME/$REPO_NAME${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify at: https://github.com/$USERNAME/$REPO_NAME"
    echo "  2. Check that it shows 🔒 Private status"
    echo "  3. Review README.md is displayed correctly"
    echo ""

    # Show clone commands
    echo "Clone this repository:"
    if [ "$AUTH_METHOD" = "SSH" ]; then
        echo -e "  ${GREEN}git clone git@github.com:$USERNAME/$REPO_NAME.git${NC}"
    else
        echo -e "  ${GREEN}git clone https://github.com/$USERNAME/$REPO_NAME.git${NC}"
    fi
    echo ""
else
    echo ""
    echo "========================================="
    echo -e "${RED}  ❌ Push Failed${NC}"
    echo "========================================="
    echo ""
    echo "Common issues:"
    if [ "$AUTH_METHOD" = "SSH" ]; then
        echo "  • SSH key not added to GitHub"
        echo "    Fix: https://github.com/settings/keys"
        echo "  • Wrong username or repository name"
        echo "  • Repository doesn't exist yet"
        echo "    Create at: https://github.com/new"
    else
        echo "  • Wrong username or password/token"
        echo "  • Token doesn't have 'repo' scope"
        echo "    Fix: https://github.com/settings/tokens"
        echo "  • Repository doesn't exist yet"
        echo "    Create at: https://github.com/new"
    fi
    echo ""
    exit 1
fi
