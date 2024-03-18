# hash:sha256:c9a38c33336e29a0186c63e967d88dd195f2d82579cdad89e5403b2f5d29c57c
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys \
        0xAD2323F17326AE31401037733E05EBFF05441C52 \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y 'deb http://deb.codeocean.com/shiny-server-trusty/ ubuntu main' \
    && apt-get purge -y --autoremove software-properties-common \
    && apt-get update \
    && apt-get install -y \
        # --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        libgit2-dev=0.28.4+dfsg.1-2 \
        libssl-dev=1.1.1f-1ubuntu2.19 \
        pandoc=2.5-3build2 \
        pkg-config=0.29.1-0ubuntu4 \
        r-base=3.6.3-2 \
        shiny-server=1.5.12.933 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    matplotlib==3.6.1 \
    nltk==3.7 \
    numpy==1.21.2 \
    pandas==1.5.1 \
    regex==2022.9.13 \
    scikit-learn==1.1.3

RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site \
    && echo 'options(Ncpus = parallel::detectCores())' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site \
    && Rscript -e 'options(warn=2); install.packages("remotes")'
RUN Rscript -e 'remotes::install_version("shiny", "1.7.3")'

# COPY postInstall /
# RUN /postInstall
