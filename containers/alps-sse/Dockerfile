FROM ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea

ARG ALPS_COMMIT=97914eba01fb8eae1b96d460b577cb62a8f7ba94
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake gfortran git \
    libboost-all-dev libhdf5-dev libopenmpi-dev libxml2-dev openmpi-bin python3 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ALPSim/ALPS.git /opt/alps-src \
    && git -C /opt/alps-src checkout "$ALPS_COMMIT"

RUN cmake -S /opt/alps-src -B /opt/alps-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/alps \
      -DALPS_BUILD_APPLICATIONS=ON -DALPS_BUILD_PYTHON=OFF \
    && cmake --build /opt/alps-build --target dirloop_sse parameter2xml convert2xml -j2 \
    && cmake --install /opt/alps-build

ENV PATH=/opt/alps/bin:$PATH
ENTRYPOINT ["dirloop_sse"]
