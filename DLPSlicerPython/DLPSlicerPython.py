import subprocess
import struct
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def call_dlpslicer(model_name, matrix, output_folder, thickness):
    exe_path = "DLPSlicerPython/DLPSlicerPython.py"
    cmd = [exe_path, "-slice"]

    # === Process parameters ===
    resx = 1920
    resy = 1080
    envx = 192.0
    envy = 108.0
    nPlacement = 1  # 0:ORIGIN, 1:ASIS, 2:CENTER
    process_params = f"{resx},{resy},{envx},{envy},{thickness},{nPlacement}"
    cmd += ["-p", process_params]

    # === Model and matrix ===
    matrix_str = ",".join(map(str, matrix))
    cmd += ["-m", model_name, matrix_str]

    # === Output folder ===
    cmd += ["-f", output_folder]

    print("Executing command:\n", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\nSTDOUT:\n", result.stdout)
    print("\nSTDERR:\n", result.stderr)

    return result

def read_detailed_binary(filepath):
    with open(filepath, "rb") as f:
        majorVersion = f.read(1).decode("ascii")
        minorVersion = f.read(1).decode("ascii")
        layerNum = struct.unpack("i", f.read(4))[0]

        data = {
            "majorVersion": majorVersion,
            "minorVersion": minorVersion,
            "layerNum": layerNum,
            "layers": []
        }

        for _ in range(layerNum):
            layer = {"contours": []}
            contourNum = struct.unpack("i", f.read(4))[0]

            for _ in range(contourNum):
                pointNum = struct.unpack("i", f.read(4))[0]
                points = [(x / 100000.0, y / 100000.0) for _ in range(pointNum) 
                  for x, y in [struct.unpack("ii", f.read(8))]]
                layer["contours"].append({"pointNum": pointNum, "points": points})

            data["layers"].append(layer)

    return data


def plot_contours_2D(data):
    for l_idx, layer in enumerate(data["layers"]):
        plt.figure(figsize=(6, 6))
        plt.title(f"Layer {l_idx + 1}")

        for contour in layer["contours"]:
            x_vals = [p[0] for p in contour["points"]]
            y_vals = [p[1] for p in contour["points"]]
            # Optional: Close the contour loop if desired
            x_vals.append(x_vals[0])
            y_vals.append(y_vals[0])
            plt.plot(x_vals, y_vals, marker='o')

        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


def plot_contours_3D(data, thickness):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Contours Across Layers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (height)")

    layer_thickness = thickness

    all_x, all_y, all_z = [], [], []

    for l_idx, layer in enumerate(data["layers"]):
        z = l_idx * layer_thickness

        for contour in layer["contours"]:
            x_vals = [p[0] for p in contour["points"]] + [contour["points"][0][0]]
            y_vals = [p[1] for p in contour["points"]] + [contour["points"][0][1]]
            z_vals = [z] * len(x_vals)
            ax.plot(x_vals, y_vals, z_vals)

            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_z.extend(z_vals)

    # === Enforce equal aspect ratio ===
    def set_equal_3d(ax, x, y, z):
        max_range = max(
            max(x) - min(x),
            max(y) - min(y),
            max(z) - min(z)
        ) / 2.0

        mid_x = (max(x) + min(x)) / 2.0
        mid_y = (max(y) + min(y)) / 2.0
        mid_z = (max(z) + min(z)) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    set_equal_3d(ax, all_x, all_y, all_z)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === Define inputs ===
    model_name = "cube.stl"
    #model_name = "cone.stl"    
    #model_name = "Gun.stl"    
    
    # transformation matrix, you can apply translation, rotation and scaling matrix to re-orient the model
    matrix = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]
    output_folder = "output"
    thickness = 0.05  # mm

    # === Call slicer ===
    call_dlpslicer(model_name, matrix, output_folder, thickness)

    # the slicer app converts the .stl file into layered images and contours (.spc file)
    # === Locate output binary ===
    model_base = os.path.splitext(os.path.basename(model_name))[0]
    spc_filename = f"{model_base}_0.spc"
    spc_path = os.path.join(output_folder, spc_filename)

    if os.path.exists(spc_path):
        data = read_detailed_binary(spc_path)
        plot_contours_3D(data, thickness)
        #plot_contours_2D(data)
    else:
        print(f"[ERROR] File not found: {spc_path}")
