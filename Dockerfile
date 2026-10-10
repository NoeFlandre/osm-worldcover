# Reproducible build environment for the dataset pipeline.
#
# The image carries the code and its pinned dependencies only. Data is never
# baked in: mount a volume at /data and point --out and --cache at it, so a
# 229 GB run writes to the host rather than into a layer.
#
# Pinned by digest so the OS and Python layer cannot change under the build.
# The tag is kept for readability. The digest is the index digest that tag
# resolved to when pinned (uv 0.9.30 on Debian bookworm, Python 3.12).
#
# Two stages share that base. The builder compiles the environment with the
# C/C++ toolchain. The runtime copies the result and adds only the GEOS shared
# library, so the toolchain and GEOS headers never reach the final image.
FROM ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/data/hf \
    PATH="/app/.venv/bin:$PATH"

FROM base AS builder

# exactextract ships no aarch64 wheel, so it is compiled from source and needs
# a C/C++ toolchain and GEOS headers. Everything else installs as a wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is cached until
# the lockfile itself changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY README.md ./
RUN uv sync --frozen --no-dev

FROM base AS runtime

# The installed packages load the GEOS shared library at run time. Only the
# library is installed here; the headers and the toolchain stay in the builder.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user. Code under /app stays root-owned and read-only.
# The pipeline writes to /data (the mounted volume) and to /app/data, which is
# the default --out and --cache location when none are given. Both are owned
# by this user.
RUN useradd --system --uid 10001 --user-group --no-create-home \
        --shell /usr/sbin/nologin owc

WORKDIR /app
COPY --from=builder /app /app

RUN install -d -o owc -g owc /data /app/data

VOLUME /data
USER owc
ENTRYPOINT ["owc"]
CMD ["--help"]
