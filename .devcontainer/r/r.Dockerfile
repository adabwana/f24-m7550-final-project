FROM rocker/r-ver:latest
# FROM r-base:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install R packages
COPY rquirements.txt /tmp/rquirements.txt
RUN Rscript -e '\
    pkg <- readLines("/tmp/rquirements.txt"); \
    pkg <- pkg[!grepl("^#", pkg) & pkg != ""]; \
    new_pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]; \
    if (length(new_pkg) > 0) install.packages(new_pkg, repos="https://cloud.r-project.org/", Ncpus=parallel::detectCores()-1); \
    invisible(sapply(pkg, require, character.only=TRUE))' 

# better plotting in vscode
# https://stackoverflow.com/questions/52284345/how-to-show-r-graph-from-visual-studio-code
RUN Rscript -e 'install.packages(c("httpgd", "languageserver"), repos="https://cloud.r-project.org/", Ncpus=parallel::detectCores()-1)'

RUN apt-get update && apt-get install -y --no-install-recommends \
    # car package dependencies
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e 'install.packages("car", dependencies=TRUE, repos="https://cloud.r-project.org/", Ncpus=parallel::detectCores()-1)'

USER $USERNAME

WORKDIR /workspace 