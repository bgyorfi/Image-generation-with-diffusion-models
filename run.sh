#!/bin/bash

function show_help {
    echo "Options:"
    echo "  --train-flowers       Train DDPM models on flowers dataset"
    echo "  --train-celebs        Train DDPM models on CelebA dataset"
    echo "  --generate-flowers    Generate images from the flowers dataset"
    echo "  --generate-celebs     Generate images from the CelebA dataset"
    echo "  --latest              Works with --generate-* to use the latest trained model, instead of the best model"    
    echo "  -h, --help            Show this help message"
}

if [ $# -eq 0 ]; then
    docker compose up
fi

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

DOCKER_COMMAND="python src/main.py ${COMMAND_ARGS[@]}"

container_name="ddpm-container"

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
