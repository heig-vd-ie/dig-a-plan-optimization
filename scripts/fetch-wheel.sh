#!/bin/bash
set -e

# Usage:
#   ./fetch_local_wheel.sh <org> <repo> <branch> [version] [dest_dir]
#
# Example:
#   ./fetch_local_wheel.sh heig-vd-ie digrid-schema main 0.5.0 ./external-dist
#   ./fetch_local_wheel.sh heig-vd-ie digrid-schema main latest ./external-dist

ORG="$1"
REPO="$2"
BRANCH="$3"
VERSION="$4"
DEST_DIR="${5:-./external-dist}"

if [[ -z "$ORG" || -z "$REPO" || -z "$BRANCH" ]]; then
    echo "Usage: $0 <org> <repo> <branch> [version|latest] [dest_dir]"
    exit 1
fi

TMP_DIR=$(mktemp -d)
mkdir -p "$DEST_DIR"

echo "Cloning $REPO ($BRANCH) into temporary folder..."
git clone --branch "$BRANCH" "https://github.com/$ORG/$REPO.git" "$TMP_DIR"

DIST_DIR="$TMP_DIR/dist"
if [[ ! -d "$DIST_DIR" ]]; then
    echo "Error: No dist folder in cloned repo"
    rm -rf "$TMP_DIR"
    exit 1
fi

# Find wheel
if [[ -z "$VERSION" || "$VERSION" == "latest" ]]; then
    echo "Finding latest version in dist/..."
    WHEEL_FILE=$(ls "$DIST_DIR"/*.whl | sort -V | tail -n1)
else
    WHEEL_FILE=$(ls "$DIST_DIR"/*"$VERSION"*.whl | head -n1)
fi

if [[ ! -f "$WHEEL_FILE" ]]; then
    echo "Error: Wheel not found for version '$VERSION'"
    rm -rf "$TMP_DIR"
    exit 1
fi

# Copy wheel to destination
cp "$WHEEL_FILE" "$DEST_DIR/"
echo "Wheel copied to $DEST_DIR/$(basename "$WHEEL_FILE")"

# Clean up
rm -rf "$TMP_DIR"
echo "Temporary clone removed."
