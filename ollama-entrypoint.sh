#!/bin/bash

# Start the Ollama server in the background
ollama serve &

# Wait a bit for the server to start
sleep 5

# Pull the LLaMA model
ollama pull llama3.2

# Wait to keep the container alive (or attach to the server)
wait
