# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
# Ensure you have a requirements.txt in the same directory as your Dockerfile
# Use pipreqs /path/to/project to generate one, if you haven't already
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
# Use this section to define any environment variables your app requires
# For example, ENV NAME World

# Run app.py when the container launches
# Include default values for required arguments or use environment variables
CMD ["python", "./src/app.py", "--num_procs", "1", "--input_file", "/app/data/input.txt", "--source_path", "/app/data/data", "--output_file", "/app/data/output.pkl"]
