from urllib.request import urlretrieve
import os
import numpy as np
import copy
import struct


class Mesh:
    def __init__(self, verts, tris):
        self.verts = verts
        self.tris = tris

    def translate(self, offset):
        result = copy.deepcopy(self)
        result.verts += offset
        return result

    def scale(self, scale):
        result = copy.deepcopy(self)
        result.verts *= scale
        return result

    def flip_normals(self):
        result = copy.deepcopy(self)
        for i, tri in enumerate(result.tris):
            result.tris[i] = [tri[2], tri[1], tri[0]]
        return result

    def write_to_cmf(self):
        result = b""
        result += struct.pack("I", len(self.tris))
        result += struct.pack("I", len(self.verts))

        for tri in self.tris:
            for i in range(3):
                result += struct.pack("i", int(tri[i]))
        for vert in self.verts:
            for i in range(3):
                result += struct.pack("f", float(vert[i]))

        return result


def main():
    mesh_url_base = "https://github.com/samuelpmish/RLUtilities/raw/develop/assets/soccar/"
    mesh_name_prefix = "soccar_"
    mesh_names = ["corner", "goal", "ramps_0", "ramps_1"]
    verts_suffix = "_vertices"
    ids_suffix = "_ids"
    extension = ".bin"

    download_dir = "./rlut_meshes/"
    os.makedirs(download_dir, exist_ok=True)

    for mesh_name in mesh_names:
        for is_verts in (True, False):
            filename = mesh_name_prefix + mesh_name + (verts_suffix if is_verts else ids_suffix) + extension
            path = os.path.join(download_dir, filename)
            if os.path.exists(path):
                print(f'Skipping "{filename}" (already exists)')
                continue

            url = mesh_url_base + filename
            print(f'Downloading "{url}"...')
            urlretrieve(url, path)

    base_meshes = {}
    for mesh_name in mesh_names:
        verts_path = os.path.join(download_dir, mesh_name_prefix + mesh_name + verts_suffix + extension)
        ids_path = os.path.join(download_dir, mesh_name_prefix + mesh_name + ids_suffix + extension)

        verts = np.fromfile(open(verts_path, "rb"), dtype=np.float32)
        ids = np.fromfile(open(ids_path, "rb"), dtype=np.uint32)

        if len(verts) % 3 != 0:
            raise ValueError(f'Wrong verts count for mesh "{verts_path}": {len(verts)}')
        if len(ids) % 3 != 0:
            raise ValueError(f'Wrong ids count for mesh "{ids_path}": {len(ids)}')

        base_meshes[mesh_name] = Mesh(verts.reshape((-1, 3)), ids.reshape((-1, 3)))

    flip_x = [-1, 1, 1]
    flip_y = [1, -1, 1]
    flip_xy = [-1, -1, 1]

    meshes = [
        base_meshes["corner"],
        base_meshes["corner"].scale(flip_x).flip_normals(),
        base_meshes["corner"].scale(flip_y).flip_normals(),
        base_meshes["corner"].scale(flip_xy),
        base_meshes["goal"].translate([0, -5120, 0]),
        base_meshes["goal"].translate([0, -5120, 0]).scale(flip_y).flip_normals(),
        base_meshes["ramps_0"],
        base_meshes["ramps_0"].scale(flip_x).flip_normals(),
        base_meshes["ramps_1"],
        base_meshes["ramps_1"].scale(flip_x).flip_normals(),
    ]

    for i in range(len(meshes)):
        meshes[i] = meshes[i].scale(1 / 50)

    output_dir = "./collision_meshes/soccar/"
    os.makedirs(output_dir, exist_ok=True)

    for i, mesh in enumerate(meshes):
        path = os.path.join(output_dir, f"mesh_{i}.cmf")
        print(f"Saving mesh to {path}...")
        with open(path, "wb") as handle:
            handle.write(mesh.write_to_cmf())

    print("Done!")


if __name__ == "__main__":
    main()
