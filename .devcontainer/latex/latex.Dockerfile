# Use a minimal debian base
FROM debian:bookworm-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    latexmk \
    imagemagick \
    ghostscript \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure ImageMagick to allow PDF and PNG operations
RUN if [ -f /etc/ImageMagick-6/policy.xml ]; then \
    # sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/rights="none" pattern="PNG"/rights="read|write" pattern="PNG"/' /etc/ImageMagick-6/policy.xml; \
    fi

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME

# Add NVIDIA GPU support
# UNCOMMENT IF YOU HAVE A GPU
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace 
