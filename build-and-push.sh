#!/bin/bash

set -e

# Extract current version from docker-compose.yml
CURRENT_VERSION=$(grep "image: julianh2o/myalpr:" docker-compose.yml | sed 's/.*:\(.*\)/\1/')
echo "Current version: $CURRENT_VERSION"

# Split version into major.minor.patch
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Increment patch version
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="${MAJOR}.${MINOR}.${NEW_PATCH}"
echo "New version: $NEW_VERSION"

# Update version in docker-compose.yml
sed -i.bak "s/julianh2o\/myalpr:${CURRENT_VERSION}/julianh2o\/myalpr:${NEW_VERSION}/" docker-compose.yml
rm docker-compose.yml.bak
echo "Updated docker-compose.yml to version $NEW_VERSION"

# Build the image for amd64
echo "Building image for linux/amd64..."
docker buildx build --platform linux/amd64 -t julianh2o/myalpr:${NEW_VERSION} -t julianh2o/myalpr:latest --load .

echo "Built image with tags:"
echo "  - julianh2o/myalpr:${NEW_VERSION}"
echo "  - julianh2o/myalpr:latest"

# Push both tags
echo "Pushing julianh2o/myalpr:${NEW_VERSION}..."
docker push julianh2o/myalpr:${NEW_VERSION}

echo "Pushing julianh2o/myalpr:latest..."
docker push julianh2o/myalpr:latest

echo "✓ Build and push complete!"
echo "Version $NEW_VERSION has been built and pushed"
