#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os

DEFAULT_SIZE = 100  # mm; size of the internal ArUco marker (without extra border)
DEFAULT_THICKNESS = 2  # mm

def generate_marker_image(dictionary, marker_id, total_grid):
    marker_size = dictionary.markerSize
    marker_code = dictionary.bytesList[marker_id]
    num_bits = marker_size * marker_size
    bits = np.unpackbits(marker_code.flatten())[-num_bits:]
    bits = bits.reshape((marker_size, marker_size))
    # Invert bits so that marker bits become black (0) and background white (1)
    bits = 1 - bits
    # Add a white border (value 1) around the marker bits.
    bordered = np.ones((marker_size + 2, marker_size + 2), dtype=np.uint8)
    bordered[1:-1, 1:-1] = bits
    if total_grid != marker_size + 2:
        bordered = cv2.resize(bordered, (total_grid, total_grid), interpolation=cv2.INTER_NEAREST)
    return bordered * 255

def cuboid_triangles(x, y, thickness, size=1):
    # Define bottom vertices (z=0)
    bl = np.array([x, y, 0])
    br = np.array([x+size, y, 0])
    tr = np.array([x+size, y+size, 0])
    tl = np.array([x, y+size, 0])
    # Define top vertices (z=thickness)
    bl_t = np.array([x, y, thickness])
    br_t = np.array([x+size, y, thickness])
    tr_t = np.array([x+size, y+size, thickness])
    tl_t = np.array([x, y+size, thickness])

    tris = []
    # Bottom face
    tris.append((bl, br, tr))
    tris.append((bl, tr, tl))
    # Top face
    tris.append((bl_t, tr_t, br_t))
    tris.append((bl_t, tl_t, tr_t))
    # Front face
    tris.append((bl, br, br_t))
    tris.append((bl, br_t, bl_t))
    # Right face
    tris.append((br, tr, tr_t))
    tris.append((br, tr_t, br_t))
    # Back face
    tris.append((tr, tl, tl_t))
    tris.append((tr, tl_t, tr_t))
    # Left face
    tris.append((tl, bl, bl_t))
    tris.append((tl, bl_t, tl_t))

    return tris

def write_stl(filename, triangles, solid_name="solid"):
    with open(filename, "w") as f:
        f.write("solid {}\n".format(solid_name))
        for tri in triangles:
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm != 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 0])
            f.write("  facet normal {} {} {}\n".format(normal[0], normal[1], normal[2]))
            f.write("    outer loop\n")
            for vertex in tri:
                f.write("      vertex {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid {}\n".format(solid_name))

def main():
    parser = argparse.ArgumentParser(
        description="Generate interleaved ArUco marker STL files (flat or stacked mode)."
    )
    parser.add_argument("marker_id", type=int, help="ArUco marker id")
    # DEFAULT_SIZE is the size for the internal ArUco marker (without extra border)
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE, help="Size of the internal ArUco marker in mm (default: 100)")
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS, help="Layer thickness in mm (default: 2)")
    parser.add_argument("--flat", action="store_true", help="Generate flat mode (both parts at same Z level). Default is stacked mode.")
    args = parser.parse_args()

    marker_id = args.marker_id

    # Try available dictionaries in order until one supports the given marker_id.
    dict_options = [
        ("4x4_50", cv2.aruco.DICT_4X4_50),
        ("4x4_100", cv2.aruco.DICT_4X4_100),
        ("4x4_250", cv2.aruco.DICT_4X4_250),
        ("4x4_1000", cv2.aruco.DICT_4X4_1000),
        ("5x5_50", cv2.aruco.DICT_5X5_50),
        ("5x5_100", cv2.aruco.DICT_5X5_100),
        ("5x5_250", cv2.aruco.DICT_5X5_250),
        ("5x5_1000", cv2.aruco.DICT_5X5_1000),
        ("6x6_50", cv2.aruco.DICT_6X6_50),
        ("6x6_100", cv2.aruco.DICT_6X6_100),
        ("6x6_250", cv2.aruco.DICT_6X6_250),
        ("6x6_1000", cv2.aruco.DICT_6X6_1000),
        ("7x7_50", cv2.aruco.DICT_7X7_50),
        ("7x7_100", cv2.aruco.DICT_7X7_100),
        ("7x7_250", cv2.aruco.DICT_7X7_250),
        ("7x7_1000", cv2.aruco.DICT_7X7_1000),
        ("aruco_original", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ]
    chosen_dict = None
    chosen_dict_name = None
    for name, dict_const in dict_options:
        d = cv2.aruco.getPredefinedDictionary(dict_const)
        if marker_id < d.bytesList.shape[0]:
            chosen_dict = d
            chosen_dict_name = name
            break

    if chosen_dict is None:
        print(f"Marker id {marker_id} is too high for available dictionaries.")
        return

    print(f"Using ArUco dictionary: {chosen_dict_name}")
    grid_dim = chosen_dict.markerSize  # inner grid size
    internal_total_grid = grid_dim + 2   # original marker grid (includes inherent white border)

    # Generate the internal marker image using the given internal grid size.
    marker_img = generate_marker_image(chosen_dict, marker_id, internal_total_grid)
    
    # Compute the cell size based on the internal marker size (args.size is the internal marker size).
    cell_size = args.size / internal_total_grid

    # Now add an extra border around the marker.
    # Since the marker output is inverted (marker pattern is black, background white),
    # we add an extra border of the opposite color: here, we use black (0).
    border_cells = 1
    new_total_grid = internal_total_grid + 2 * border_cells
    new_marker_img = np.zeros((new_total_grid, new_total_grid), dtype=np.uint8)
    new_marker_img[border_cells:border_cells+internal_total_grid, border_cells:border_cells+internal_total_grid] = marker_img
    marker_img = new_marker_img
    total_grid = new_total_grid

    # The overall printed size will exceed args.size by the border.
    overall_size = args.size + 2 * border_cells * cell_size

    out_dir = f"aruco-stl-{marker_id}"
    os.makedirs(out_dir, exist_ok=True)

    if args.flat:
        # Flat mode: both parts at same Z level.
        white_tris = []
        black_tris = []
        for row in range(total_grid):
            for col in range(total_grid):
                x = col * cell_size
                y = (total_grid - 1 - row) * cell_size
                if marker_img[row, col] > 128:
                    white_tris.extend(cuboid_triangles(x, y, args.thickness, size=cell_size))
                else:
                    black_tris.extend(cuboid_triangles(x, y, args.thickness, size=cell_size))
        # Keep inversion swap as before.
        white_filename = os.path.join(out_dir, "black.stl")
        black_filename = os.path.join(out_dir, "white.stl")
        write_stl(white_filename, white_tris, solid_name="white")
        write_stl(black_filename, black_tris, solid_name="black")
        print(f"Generated flat mode STL files:\n  {white_filename}\n  {black_filename}")
    else:
        # Stacked mode: full base and top layer offset in Z.
        # Base: full board with overall size (including extra border)
        base_tris = cuboid_triangles(0, 0, args.thickness, size=overall_size)
        # Top: only marker pattern cells (black) offset in Z by the base thickness.
        top_tris = []
        for row in range(total_grid):
            for col in range(total_grid):
                if marker_img[row, col] <= 128:
                    x = col * cell_size
                    y = (total_grid - 1 - row) * cell_size
                    cell_tris = cuboid_triangles(x, y, args.thickness, size=cell_size)
                    # Offset the top layer in Z by the base thickness.
                    cell_tris = [tuple(np.array(vertex) + np.array([0, 0, args.thickness]) for vertex in tri) for tri in cell_tris]
                    top_tris.extend(cell_tris)
        base_filename = os.path.join(out_dir, "base.stl")
        top_filename = os.path.join(out_dir, "top.stl")
        write_stl(base_filename, base_tris, solid_name="base")
        write_stl(top_filename, top_tris, solid_name="top")
        print(f"Generated stacked mode STL files:\n  {base_filename}\n  {top_filename}")

if __name__ == "__main__":
    main()
