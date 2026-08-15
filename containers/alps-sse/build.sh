#!/usr/bin/env bash
set -euo pipefail

engine=${CONTAINER_ENGINE:-docker}
image=${ALPS_IMAGE:-pmrqmc/alps-dirloop-sse:v2.3.4}
root=$(cd "$(dirname "$0")/../.." && pwd)

"$engine" build --pull --build-arg \
  ALPS_COMMIT=97914eba01fb8eae1b96d460b577cb62a8f7ba94 \
  -t "$image" "$root/containers/alps-sse"
"$engine" image inspect "$image" > "$root/containers/alps-sse/image-inspect.json"
"$engine" image inspect --format '{{json .RepoDigests}}' "$image"
