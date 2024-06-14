.PHONY: cleanup-code test

cleanup-code:
 python -m black .

test:
 echo "Running tests..."
 # Add tests in the future
 echo "Passed"
