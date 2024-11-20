# Base Layer
FROM clojure:temurin-21-tools-deps-bookworm-slim

# Architecture detection for conditional installations
ARG TARGETARCH=amd64

# Common System Dependencies Layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bash \
    openssh-client \
    openssh-server \
    sudo \
    libx11-dev \
    libxi6 \
    libxtst6 \
    libxrender1 \
    libxext6 \
    libxft2 \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Quarto Layer - Automatic architecture detection
RUN ARCH=$(dpkg --print-architecture) \
    && wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.450/quarto-1.3.450-linux-${ARCH}.deb \
    && dpkg -i quarto-1.3.450-linux-${ARCH}.deb \
    && rm quarto-1.3.450-linux-${ARCH}.deb \
    && apt-get update \
    && apt-get install -f -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python Layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    python3-pandas \
    python3-ipykernel \
    python3-matplotlib \
    python3-seaborn \
    python3-scipy \
    python3-sklearn \
    python3-statsmodels \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# R Layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    littler \
    r-cran-littler \
    r-cran-docopt \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgit2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/R/site-library/littler/examples/install2.r /usr/local/bin/install2.r \
    && ln -s /usr/lib/R/site-library/littler/examples/installGithub.r /usr/local/bin/installGithub.r

# Install R packages from requirements file
COPY /rquirements.txt /tmp/rquirements.txt
RUN Rscript -e '\
    pkg <- readLines("/tmp/rquirements.txt"); \
    pkg <- pkg[!grepl("^#", pkg) & pkg != ""]; \
    new_pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]; \
    if (length(new_pkg) > 0) install.packages(new_pkg, repos="https://cloud.r-project.org/", Ncpus=parallel::detectCores()-1); \
    invisible(sapply(pkg, require, character.only=TRUE))' \
    && rm -rf /tmp/downloaded_packages /tmp/rquirements.txt

# User Setup Layer
RUN useradd -ms /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Environment Configuration Layer
WORKDIR /workspace/ds-in-clojure
ENV PATH="/usr/local/bin:/usr/bin:/usr/local/bin/python3:${PATH}"
ENV DISPLAY=host.docker.internal:0
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV JAVA_TOOL_OPTIONS="-Djava.awt.headless=true"

# Optional GPU Support (commented out)
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Switch to non-root user
USER vscode