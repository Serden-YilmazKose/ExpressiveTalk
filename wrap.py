import cv2
import numpy as np

def piecewise_affine_warp(img, src_pts, dst_pts, preserve_mask=None):
    """
    Triangulate on src_pts (Delaunay), warp each triangle to dst.
    Optionally preserve a binary mask area (e.g., lips region).
    """
    h, w = img.shape[:2]
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for (x,y) in src_pts:
        subdiv.insert((float(x), float(y)))
    triangles = subdiv.getTriangleList().astype(np.float32).reshape(-1,3,2)

    out = np.zeros_like(img)
    accum_mask = np.zeros((h,w), np.uint8)

    for tri in triangles:
        src_tri = tri
        idx = [np.argmin(np.linalg.norm(src_pts - v, axis=1)) for v in tri]
        dst_tri = dst_pts[idx]

        r1 = cv2.boundingRect(np.float32([src_tri]))
        r2 = cv2.boundingRect(np.float32([dst_tri]))

        src_offset = np.float32([[p[0]-r1[0], p[1]-r1[1]] for p in src_tri])
        dst_offset = np.float32([[p[0]-r2[0], p[1]-r2[1]] for p in dst_tri])

        M = cv2.getAffineTransform(src_offset, dst_offset)
        patch = img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        warped = cv2.warpAffine(patch, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_offset), 255)

        roi = out[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
        roi_mask = accum_mask[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

        roi[mask>0] = warped[mask>0]
        roi_mask[mask>0] = 255

    # Blend with original to preserve lips region if given
    if preserve_mask is not None:
        preserve = cv2.bitwise_and(img, img, mask=preserve_mask)
        inv = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(preserve_mask))
        out = cv2.add(inv, preserve)

    return out
