#!/bin/bash
# setup_remote.sh - Script to set up a remote GitHub repository connection
# Usage: ./setup_remote.sh <github_username> <repository_name>

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments.${NC}"
    echo "Usage: ./setup_remote.sh <github_username> <repository_name>"
    echo "Example: ./setup_remote.sh yourusername healthcare-rag"
    exit 1
fi

GITHUB_USERNAME="$1"
REPO_NAME="$2"
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo -e "${BLUE}Setting up remote repository connection...${NC}"
echo "GitHub Username: $GITHUB_USERNAME"
echo "Repository Name: $REPO_NAME"
echo "Remote URL: $REMOTE_URL"

# Check if remote 'origin' already exists
if git remote | grep -q "^origin$"; then
    echo -e "${YELLOW}Remote 'origin' already exists. Do you want to update it? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        git remote remove origin
        git remote add origin "$REMOTE_URL"
        echo -e "${GREEN}Updated remote 'origin' to $REMOTE_URL${NC}"
    else
        echo -e "${YELLOW}Keeping existing remote 'origin'.${NC}"
    fi
else
    git remote add origin "$REMOTE_URL"
    echo -e "${GREEN}Added remote 'origin' as $REMOTE_URL${NC}"
fi

# Push branches to remote
echo -e "${BLUE}Do you want to push all branches to remote? (y/n)${NC}"
read -r push_all
if [[ "$push_all" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${BLUE}Pushing all branches...${NC}"
    
    # Get current branch to return to it later
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    
    # Push main branch
    git checkout main
    git push -u origin main
    
    # Push develop branch
    git checkout develop
    git push -u origin develop
    
    # Push feature branches
    for branch in $(git branch | grep "feature/" | tr -d " "); do
        git checkout "$branch"
        git push -u origin "$branch"
    done
    
    # Return to original branch
    git checkout "$CURRENT_BRANCH"
    
    echo -e "${GREEN}All branches pushed to remote.${NC}"
else
    echo -e "${BLUE}Pushing current branch only...${NC}"
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    git push -u origin "$CURRENT_BRANCH"
    echo -e "${GREEN}Branch '$CURRENT_BRANCH' pushed to remote.${NC}"
fi

echo -e "${BLUE}Remote repository setup complete!${NC}"
echo -e "${YELLOW}Repository URL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}${NC}"
