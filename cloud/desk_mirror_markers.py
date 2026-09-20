#!/usr/bin/env python3
"""On forlindesk2: write the mirror.json coverage markers of the Paris 4 CT mirror: levels 1-9 are complete,
level 0 is known only inside the teacher boxes (~/boxes_p4.txt, "z y x Z Y X" lines)."""
import json, os, sys
B = "/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr"
for l in range(1, 10):
    if os.path.isdir(f"{B}/{l}"):
        json.dump({"complete": True}, open(f"{B}/{l}/mirror.json", "w"))
bx = [[int(q) for q in line.split()] for line in open(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/boxes_p4.txt")) if len(line.split()) == 6]
json.dump({"boxes": bx}, open(f"{B}/0/mirror.json", "w"))
print(f"markers written: levels 1-9 complete, level 0 {len(bx)} boxes")
