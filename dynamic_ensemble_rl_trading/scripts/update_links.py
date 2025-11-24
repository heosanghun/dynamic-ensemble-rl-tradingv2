"""
Script to update anonymous repository links in project files.

Usage:
    python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
    python scripts/update_links.py --github https://github.com/username/repo
"""

import sys
import re
from pathlib import Path
from typing import Optional


def update_4open_link(link_id: str):
    """Update 4open.science anonymous link."""
    # Remove 'r/' prefix if present
    if link_id.startswith('r/'):
        link_id = link_id[2:]
    full_link = f"https://anonymous.4open.science/r/{link_id}"
    return full_link, "YOUR-ANONYMOUS-LINK-ID"


def update_github_link(github_url: str):
    """Update GitHub repository link."""
    return github_url, "YOUR-ANONYMOUS-LINK-ID"


def update_file(file_path: Path, old_pattern: str, new_link: str):
    """Update links in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Replace the pattern
        if old_pattern in content:
            content = content.replace(old_pattern, new_link)
            file_path.write_text(content, encoding='utf-8')
            print(f"Updated: {file_path}")
            return True
        else:
            print(f"No pattern found in: {file_path}")
            return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID")
        print("  python scripts/update_links.py --github https://github.com/username/repo")
        sys.exit(1)
    
    # Parse arguments
    if sys.argv[1] == "--github":
        if len(sys.argv) < 3:
            print("Error: GitHub URL required")
            sys.exit(1)
        new_link, old_pattern = update_github_link(sys.argv[2])
    else:
        link_id = sys.argv[1]
        new_link, old_pattern = update_4open_link(link_id)
    
    print(f"Updating links from '{old_pattern}' to '{new_link}'")
    print("-" * 60)
    
    # Files to update
    project_root = Path(__file__).parent.parent
    files_to_update = [
        project_root / "README.md",
        project_root / "setup.py",
        project_root / "docs" / "REPRODUCTION.md",
    ]
    
    updated_count = 0
    for file_path in files_to_update:
        if file_path.exists():
            if update_file(file_path, old_pattern, new_link):
                updated_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print("-" * 60)
    print(f"Updated {updated_count} file(s)")
    print(f"New link: {new_link}")
    print("\nPlease verify the changes and commit if needed.")


if __name__ == '__main__':
    main()

