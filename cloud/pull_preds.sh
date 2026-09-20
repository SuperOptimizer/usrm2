#!/bin/bash
# On forlindesk2: mirror the exported upstream prediction pyramids (volcomp mask mode, levels named by micron) from the
# public tree into the local volcomp mirror, so training reads them locally (never over HTTP).
#   cloud/pull_preds.sh [SCROLL]      default PHercParis4; log to stdout. Re-runnable (wget -N skips unchanged files).
S=${1:-PHercParis4}
SRC="https://dl.ash2txt.org/community-uploads/forrest/volcomp/$S/representations/predictions/surfaces/"
DST="/vesuvius/usrm/volcomp/$S/representations/predictions/surfaces"
mkdir -p "$DST" && cd "$DST" || exit 1
# community-uploads/forrest/volcomp/<S>/representations/predictions/surfaces/ = 7 path components to cut
wget -q -r -l inf -np -nH -N --cut-dirs=7 -R "index.html*" -e robots=off "$SRC"
echo "pulled $(find . -type f | wc -l) files, $(du -sh . | cut -f1) into $DST $(date)"
