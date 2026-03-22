#!/bin/bash
# Patch frozendict for Python 3.10+ compatibility (collections.abc.Mapping)
FROZENDICT_INIT="/usr/local/lib/python3.11/dist-packages/frozendict/__init__.py"

if [ -f "$FROZENDICT_INIT" ]; then
  if ! grep -q "import collections.abc" "$FROZENDICT_INIT"; then
    echo "[PATCH] Fixing frozendict for Python 3.10+..."
    sed -i '1s/^/import collections.abc\ncollections.Mapping = collections.abc.Mapping\ncollections.MutableMapping = collections.abc.MutableMapping\n/' "$FROZENDICT_INIT"
    echo "[PATCH] frozendict patched successfully"
  else
    echo "[SKIP] frozendict already patched"
  fi
else
  echo "[WARN] frozendict __init__.py not found at $FROZENDICT_INIT"
fi

exec "$@"
