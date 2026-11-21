#!/usr/bin/env bash
# Builds everything Part B needs into ./deps and a conda env called xfer-mpc.
#   bash environment/setup_mpc.sh && source deps/env.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEPS=$ROOT/deps
ACADOS_COMMIT=03475431cb8f34f478458671b99a45e6f3a893bf      # acados v0.4.3-16, as used in Jan 2025
GYMQ_COMMIT=1b5b6c9f94a0d9eb10fd1d40e1e493f698577ac7        # iit-DLSLab/gym-quadruped, 2025-01-06
mkdir -p "$DEPS"

conda create -y -q -n xfer-mpc -c conda-forge python=3.11 pip "cmake<4" make compilers ffmpeg >/dev/null
PY=$(conda run -n xfer-mpc which python)
PIP="$PY -m pip"
CMAKE=$(dirname "$PY")/cmake

# acados (C library + python template + tera renderer)
if [ ! -d "$DEPS/acados" ]; then
  git clone -q https://github.com/acados/acados.git "$DEPS/acados"
  git -C "$DEPS/acados" checkout -q $ACADOS_COMMIT
  git -C "$DEPS/acados" submodule update --init --recursive -q
fi
$CMAKE -S "$DEPS/acados" -B "$DEPS/acados/build" -DACADOS_WITH_QPOASES=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
$CMAKE --build "$DEPS/acados/build" -j"$(nproc)" --target install >/dev/null
mkdir -p "$DEPS/acados/bin"
curl -sL -o "$DEPS/acados/bin/t_renderer" \
  https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux
chmod +x "$DEPS/acados/bin/t_renderer"
$PIP install -q -e "$DEPS/acados/interfaces/acados_template"

# python deps
$PIP install -q -r "$ROOT/environment/requirements-mpc.txt"

# gym-quadruped (upstream, pinned) + the extra robot models used in this project
if [ ! -d "$DEPS/gym-quadruped" ]; then
  git clone -q https://github.com/iit-DLSLab/gym-quadruped.git "$DEPS/gym-quadruped"
  git -C "$DEPS/gym-quadruped" checkout -q $GYMQ_COMMIT
fi
cp -r "$ROOT/mpc_transfer/extra_robot_models/." "$DEPS/gym-quadruped/gym_quadruped/robot_model/"
$PIP install -q -e "$DEPS/gym-quadruped" --no-deps
$PIP install -q -e "$ROOT/mpc_transfer/Quadruped-PyMPC" --no-deps

cat > "$DEPS/env.sh" <<EOT
conda activate xfer-mpc 2>/dev/null || export PATH=$(dirname "$PY"):\$PATH
export ACADOS_SOURCE_DIR=$DEPS/acados
export LD_LIBRARY_PATH=$DEPS/acados/lib:\${LD_LIBRARY_PATH:-}
export MUJOCO_GL=\${MUJOCO_GL:-egl}
EOT
echo "done. run: source deps/env.sh"
