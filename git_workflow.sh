#!/bin/bash
# git_workflow.sh - Helper script for Git operations in the Healthcare RAG project
# Usage: ./git_workflow.sh [operation] [arguments]

set -e  # Exit on any error

# Configuration
MAIN_BRANCH="main"
DEVELOP_BRANCH="develop"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed.${NC}"
    exit 1
fi

# Function to display help
show_help() {
    echo -e "${BLUE}Healthcare RAG Git Workflow Helper${NC}"
    echo "Usage: ./git_workflow.sh [operation] [arguments]"
    echo ""
    echo "Operations:"
    echo "  init              - Initialize repository and create main branches"
    echo "  feature <name>    - Create and switch to a new feature branch"
    echo "  bugfix <name>     - Create and switch to a new bugfix branch"
    echo "  commit <message>  - Stage all changes and commit with message"
    echo "  push              - Push current branch to remote"
    echo "  status            - Show git status"
    echo "  done              - Finish current feature (commits, pushes, and creates PR message)"
    echo "  sync              - Pull latest changes from develop branch"
    echo "  help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./git_workflow.sh feature add-vector-search"
    echo "  ./git_workflow.sh commit \"Add vector search functionality\""
    echo "  ./git_workflow.sh done"
}

# Initialize repository
init_repo() {
    echo -e "${BLUE}Initializing Git repository...${NC}"
    
    # Check if git is already initialized
    if [ -d .git ]; then
        echo -e "${YELLOW}Git repository already initialized.${NC}"
    else
        git init
        echo -e "${GREEN}Git repository initialized.${NC}"
    fi
    
    # Create main branch
    git checkout -b $MAIN_BRANCH 2>/dev/null || git checkout $MAIN_BRANCH
    
    # Initial commit if no commits exist
    if [ -z "$(git log -1 2>/dev/null)" ]; then
        echo -e "${BLUE}Creating initial commit...${NC}"
        git add .
        git commit -m "Initial commit: Project structure"
        echo -e "${GREEN}Initial commit created.${NC}"
    fi
    
    # Create develop branch
    git checkout -b $DEVELOP_BRANCH 2>/dev/null || git checkout $DEVELOP_BRANCH
    
    echo -e "${GREEN}Repository initialized with $MAIN_BRANCH and $DEVELOP_BRANCH branches.${NC}"
}

# Create a new feature branch
create_feature() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Feature name is required.${NC}"
        echo "Usage: ./git_workflow.sh feature <name>"
        exit 1
    fi
    
    FEATURE_NAME="feature/$1"
    echo -e "${BLUE}Creating feature branch: $FEATURE_NAME${NC}"
    
    # Make sure we're on develop first
    git checkout $DEVELOP_BRANCH
    
    # Create feature branch
    git checkout -b $FEATURE_NAME
    
    echo -e "${GREEN}Created and switched to feature branch: $FEATURE_NAME${NC}"
}

# Create a new bugfix branch
create_bugfix() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Bugfix name is required.${NC}"
        echo "Usage: ./git_workflow.sh bugfix <name>"
        exit 1
    fi
    
    BUGFIX_NAME="bugfix/$1"
    echo -e "${BLUE}Creating bugfix branch: $BUGFIX_NAME${NC}"
    
    # Make sure we're on develop first
    git checkout $DEVELOP_BRANCH
    
    # Create bugfix branch
    git checkout -b $BUGFIX_NAME
    
    echo -e "${GREEN}Created and switched to bugfix branch: $BUGFIX_NAME${NC}"
}

# Commit changes
commit_changes() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Commit message is required.${NC}"
        echo "Usage: ./git_workflow.sh commit \"Your commit message\""
        exit 1
    fi
    
    echo -e "${BLUE}Committing changes: $1${NC}"
    
    # Stage all changes
    git add .
    
    # Commit with message
    git commit -m "$1"
    
    echo -e "${GREEN}Changes committed.${NC}"
}

# Push to remote
push_branch() {
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    echo -e "${BLUE}Pushing $CURRENT_BRANCH to remote...${NC}"
    
    git push -u origin $CURRENT_BRANCH
    
    echo -e "${GREEN}Branch $CURRENT_BRANCH pushed to remote.${NC}"
}

# Finish feature
finish_feature() {
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    
    if [[ ! $CURRENT_BRANCH =~ ^feature/ && ! $CURRENT_BRANCH =~ ^bugfix/ ]]; then
        echo -e "${RED}Error: Not on a feature or bugfix branch.${NC}"
        echo "Current branch: $CURRENT_BRANCH"
        exit 1
    fi
    
    echo -e "${BLUE}Finishing branch: $CURRENT_BRANCH${NC}"
    
    # Check if there are uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}You have uncommitted changes. Would you like to commit them? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo "Enter commit message:"
            read -r commit_msg
            git add .
            git commit -m "$commit_msg"
        fi
    fi
    
    # Push to remote
    git push -u origin $CURRENT_BRANCH
    
    echo -e "${GREEN}Branch $CURRENT_BRANCH has been pushed to remote.${NC}"
    echo -e "${YELLOW}Please create a pull request on GitHub to merge into develop.${NC}"
    
    # Generate PR template
    echo ""
    echo -e "${BLUE}Pull Request Template:${NC}"
    echo "----------------------------------------"
    echo "## Description"
    echo "$(echo $CURRENT_BRANCH | sed 's/feature\///' | sed 's/bugfix\///' | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')"
    echo ""
    echo "## Changes Made"
    echo "- "
    echo ""
    echo "## Testing"
    echo "- [ ] Unit tests"
    echo "- [ ] Integration tests"
    echo "- [ ] Manual testing"
    echo ""
    echo "## Notes"
    echo ""
    echo "----------------------------------------"
}

# Sync with develop
sync_with_develop() {
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    
    echo -e "${BLUE}Syncing with develop branch...${NC}"
    
    # Save current branch
    git checkout $DEVELOP_BRANCH
    git pull origin $DEVELOP_BRANCH
    git checkout $CURRENT_BRANCH
    git merge $DEVELOP_BRANCH
    
    echo -e "${GREEN}Synced $CURRENT_BRANCH with latest changes from $DEVELOP_BRANCH.${NC}"
}

# Main script logic
case "$1" in
    "init")
        init_repo
        ;;
    "feature")
        create_feature "$2"
        ;;
    "bugfix")
        create_bugfix "$2"
        ;;
    "commit")
        commit_changes "$2"
        ;;
    "push")
        push_branch
        ;;
    "status")
        git status
        ;;
    "done")
        finish_feature
        ;;
    "sync")
        sync_with_develop
        ;;
    "help" | *)
        show_help
        ;;
esac

exit 0
