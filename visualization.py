import os
import numpy as np

def mat_to_quaternion(R):
    q = np.zeros(4)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2,1] - R[1,2]) / s
        q[2] = (R[0,2] - R[2,0]) / s
        q[3] = (R[1,0] - R[0,1]) / s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2]) * 2
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2]) * 2
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s
    return q

def save_as_colmap(points3D, obs, P_list, K_list, image_names, points_file, outdir="colmap_sfm"):
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # 1. cameras.txt
    # -------------------------
    with open(f"{outdir}/cameras.txt","w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, K in enumerate(K_list, start=1):
            fx = K[0,0]; fy = K[1,1]
            cx = K[0,2]; cy = K[1,2]
            # 假设图像大小未知，用常见尺寸
            f.write(f"{i} PINHOLE 1920 1080 {fx} {fy} {cx} {cy}\n")

    # -------------------------
    # 2. images.txt
    # -------------------------
    with open(f"{outdir}/images.txt","w") as f:
        f.write("# IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME\n")

        for i, P in enumerate(P_list, start=1):

            K_inv = np.linalg.inv(K_list[i-1])
            Rt = K_inv @ P
            R = Rt[:,:3]
            t = Rt[:,3]

            # 转换为世界到相机
            R_wc = R.T
            t_wc = -R.T @ t

            q = mat_to_quaternion(R_wc)

            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t_wc[0]} {t_wc[1]} {t_wc[2]} {i} {image_names[i-1]}\n")
            f.write("\n")   # COLMAP 需要一个空行

    # -------------------------
    # 3. points3D.txt
    # -------------------------
    with open(points_file) as f:
        lines = f.readlines()

    with open(f"{outdir}/points3D.txt","w") as f:
        f.write("# POINT3D_ID, X Y Z, R G B, ERROR, TRACK[]\n")

        for pt_id, line in enumerate(lines):
            vals = line.strip().split()
            X,Y,Z = vals[1:4]
            R,G,B = vals[4:7]

            # 构造 tracks
            track_elems = []
            for (pidx, img, corner) in obs:
                if pidx == pt_id:
                    track_elems.append(f"{img+1} {corner}")

            track_str = " ".join(track_elems)

            f.write(f"{pt_id} {X} {Y} {Z} {R} {G} {B} 1.0 {track_str}\n")
