# Dockerfile for Predictive Occupancy Modeling
#
# This image installs the package in editable mode along with its
# dependencies.  It also installs Jupyter Lab for interactive
# exploration of the notebooks.  When run without arguments the
# container will execute the ``predictive-occ-modeling`` CLI and
# display help.  To start a Jupyter server override the default
# command as shown in the repository README.

FROM python:latest-slim

# Install system packages required for building some Python wheels
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Upgrade pip and install the project in editable mode
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . 

# Expose the default Jupyter port
EXPOSE 8888

# The default entrypoint prints CLI help.  Override CMD at runtime to
# run alternative commands (e.g. ``jupyter lab``).
ENTRYPOINT ["predictive-occ-modeling"]
CMD ["--help"]