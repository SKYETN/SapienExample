# import trimesh
# import os

# base_path = "/home/shiqintong/Downloads/wheelchair_description/urdf/meshes/grippers/robotiq_2f_85/collision"
# for root, _, files in os.walk(base_path):
#     for file in files:
#         if file.lower().endswith(".stl") and not file.endswith(".convex.stl"):
#             path = os.path.join(root, file)
#             try:
#                 mesh = trimesh.load_mesh(path)
#                 if mesh.is_empty:
#                     print(f"❌ {file}: mesh is empty, skipping")
#                     continue
#                 convex = mesh.convex_hull
#                 out_path = path + ".convex.stl"
#                 print(f"✅ Generating: {out_path}")
#                 convex.export(out_path)
#             except Exception as e:
#                 print(f"❌ Failed to process {file}: {e}")

import trimesh
import os

base_path = "/home/shiqintong/Downloads/wheelchair_description/urdf/meshes/grippers/robotiq_2f_85/collision"

for root, _, files in os.walk(base_path):
    for file in files:
        if file.lower().endswith((".stl", ".dae")) and not file.endswith(".convex.stl"):
            path = os.path.join(root, file)
            out_path = path + ".convex.stl"
            try:
                mesh = trimesh.load_mesh(path, force='mesh')
                if mesh.is_empty:
                    print(f"❌ {file}: Empty mesh, skipping")
                    continue
                convex = mesh.convex_hull
                print(f"✅ Generating: {out_path}")
                convex.export(out_path)
            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")
