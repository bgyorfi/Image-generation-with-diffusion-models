#!/bin/bash

# Function to display help
function show_help {
    echo "Usage: $0 [--train] [--flowers-only] [--eval] [--generate-flowers] [--generate-celebs]"
    echo "Options:"
    echo "  --train-flowers       Train DDPM models on flowers dataset"
    echo "  --train-celebs        Train DDPM models on CelebA dataset"
    echo "  --eval                Evaluate DDPM models"
    echo "  --generate-flowers    Generate images from the flowers dataset"
    echo "  --generate-celebs     Generate images from the CelebA dataset"
    echo "  --latest              Works with --eval or --generate-* to use the latest trained model, instead of the best model"    
    echo "  -h, --help            Show this help message"
}

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Parse command-line arguments
COMMAND_ARGS=()
for arg in "$@"; do
    case $arg in
        --train-flowers)
            COMMAND_ARGS+=("--train-flowers")
            shift
            ;;
        --train-celebs)
            COMMAND_ARGS+=("--train-celebs")
            shift
            ;;
        --eval)
            COMMAND_ARGS+=("--eval")
            shift
            ;;
        --generate-flowers)
            COMMAND_ARGS+=("--generate-flowers")
            shift
            ;;
        --generate-celebs)
            COMMAND_ARGS+=("--generate-celebs")
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            show_help
            exit 1
            ;;
    esac
done

# Build the command to run inside the container
DOCKER_COMMAND="python src/main.py ${COMMAND_ARGS[@]}"

# Define the container name
container_name="ddpm-container"

# Check if the container exists (running or stopped)
if docker ps -a --filter "name=${container_name}" | grep -q ${container_name}; then
    if docker ps --filter "name=${container_name}" | grep -q ${container_name}; then
        docker exec -it ${container_name} $DOCKER_COMMAND
    else
        docker compose up -d
        docker exec -it ${container_name} $DOCKER_COMMAND
    fi
else
    docker compose up -d
    docker exec -it ${container_name} $DOCKER_COMMAND
fi
