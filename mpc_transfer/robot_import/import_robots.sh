#!/usr/bin/env bash
# Regenerates the extra robot models from example-robot-data (conda install -c conda-forge example-robot-data).
# ERD=<prefix>/share/example-robot-data/robots   (defaults to the xfer-trajopt conda env)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ERD=${ERD:-$(conda run -n xfer-trajopt python -c "import sys;print(sys.prefix)")/share/example-robot-data/robots}
OUT=$HERE/../extra_robot_models
conv() { python "$HERE/urdf_to_gymquadruped.py" --name "$1" --urdf "$ERD/$2" ${3:+--srdf "$ERD/$3"} --out "$OUT" ${4:-}; }
conv a1       a1_description/urdf/a1.urdf                     a1_description/srdf/a1.srdf
conv laikago  laikago_description/urdf/laikago.urdf
# Solo-12: the URDF effort (1000 N·m) is a placeholder; ODRI motors give ~2.7 N·m (Grimminger et al., RA-L 2020)
conv solo12   solo_description/robots/solo12.urdf             solo_description/srdf/solo.srdf "--effort 2.7"
conv anymal_b anymal_b_simple_description/robots/anymal.urdf  anymal_b_simple_description/srdf/anymal.srdf
conv anymal_c anymal_c_simple_description/urdf/anymal.urdf    anymal_c_simple_description/srdf/anymal.srdf
conv b1       b1_description/urdf/b1.urdf                     b1_description/srdf/b1.srdf
conv hyq      hyq_description/robots/hyq_no_sensors.urdf      hyq_description/srdf/hyq.srdf
