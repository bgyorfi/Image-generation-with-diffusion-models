#!/bin/bash

function show_help {
    echo "Usage: $0 [--train] [--evaluate] [--generate]"
    echo "Options:"
    echo "  --train       Train the model"
    echo "  --evaluate    Evaluate the model"
    echo "  --generate    Generate images"
    echo "  --all         Run train, evaluate, and generate sequentially"
    echo "  -h, --help    Show this help message"
}

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND_ARGS=()
for arg in "$@"; do
    case $arg in
        --train)
            COMMAND_ARGS+=("--train")
            shift
            ;;
        --evaluate)
            COMMAND_ARGS+=("--evaluate")
            shift
            ;;
        --generate)
            COMMAND_ARGS+=("--generate")
            shift
            ;;
        --all)
            COMMAND_ARGS+=("--train" "--evaluate" "--generate")
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

DOCKER_COMMAND="python main.py ${COMMAND_ARGS[@]}"

docker-compose run ddpm-container $DOCKER_COMMAND
