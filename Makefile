build:\n\t@echo "Building Docker image..."
\tdocker build -t full-stack-cv-project .

run:\n\t@echo "Running container..."
\tdocker run -p 5000:5000 --rm full-stack-cv-project

build-run:\n\t@echo "Building and running..."
\tdocker build -t full-stack-cv-project . && docker run -p 5000:5000 --rm full-stack-cv-project

clean:\n\t@echo "Removing Docker container and image..."
\t@docker stop $(docker ps -aq --filter name=full-stack-cv-project) 2>/dev/null || true
\t@docker rmi full-stack-cv-project 2>/dev/null || true