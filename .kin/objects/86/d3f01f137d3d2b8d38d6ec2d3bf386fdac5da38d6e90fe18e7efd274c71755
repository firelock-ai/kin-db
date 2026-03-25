FROM rust:1.82-slim AS builder
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN cargo build --release

FROM rust:1.82-slim AS test-runner
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
CMD ["cargo", "test"]
