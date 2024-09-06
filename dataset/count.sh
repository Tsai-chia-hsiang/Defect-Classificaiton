r="${1:-.}"

# Iterate through all subdirectories in the specified directory
for dir in "$r"/* ; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Print the directory name without the trailing slash
        echo "${dir%/}: $(ls "$dir" | wc -l)"
    fi
done