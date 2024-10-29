import cv2
import numpy as np

obj_pts = [
    [100, 285],             # GL0
    [140, 285],             # GL1
    [140, 415],             # GL2
    [100, 415],             # GL3
    [100, 210],             # OB0
    [100, 50]               # P0
]

scene_pts = [
    [731, 344],
    [1029, 367],
    [361, 537],
    [28, 507],
    [1062, 268],
    [1400, 188]
]

mp = (70, 210)

def rescale_frame(frame_input, percent=75):    
    width = int(frame_input.shape[1] * percent / 100)    
    height = int(frame_input.shape[0] * percent / 100)    
    dim = (width, height)    
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_CUBIC)

def get_advert_coords_from_midpt(ad_im, midpt):
    axh = ad_im.shape[0] // 2
    ayh = ad_im.shape[1] // 2

    return np.array([
        [midpt[0] - axh, midpt[1] - ayh],
        [midpt[0] + axh, midpt[1] - ayh],
        [midpt[0] + axh, midpt[1] + ayh],
        [midpt[0] - axh, midpt[1] + ayh],
    ])

def mark_pts(im, pts, marker_color = (255, 50, 0), text_color = (0, 50, 255), marker_type = cv2.MARKER_CROSS, text_sz = 2, marker_sz = 30):
    canvas = im.copy()
    for i, p in enumerate(pts):
        cv2.drawMarker(canvas, (int(p[0]), int(p[1])), marker_color, marker_type, marker_sz, 3)
        cv2.putText(canvas, str(i), (int(p[0]), int(p[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, text_sz, text_color, 2)
    return canvas


def perspective_tx_cv(pts, h):
    pts_reshaped = np.float32(pts)[:, np.newaxis, :]
    return cv2.perspectiveTransform(pts_reshaped, h).squeeze(1)


def warp_whole_image(image, H):
    height, width = image.shape[:2]
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    transformed_corners = cv2.perspectiveTransform(np.array([src_points]), H)[0]

    x_coords, y_coords = zip(*transformed_corners)
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))

    new_width = max_x - min_x
    new_height = max_y - min_y
    translation_matrix = np.array([[1, 0, -min_x],
                                [0, 1, -min_y],
                                [0, 0, 1]])
    adjusted_homography = translation_matrix @ H

    warped_image = cv2.warpPerspective(image, adjusted_homography, (new_width, new_height))
    return warped_image

def add_perspective(source_image, destination_image, dst_pts):
    src_height, src_width = source_image.shape[:2]
    src_points = np.float32([[0, 0], [src_width, 0], [src_width, src_height], [0, src_height]])
    dst_points = np.float32([dst_pts[3], dst_pts[0], dst_pts[1], dst_pts[2]])

    homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(source_image, homography_matrix, (destination_image.shape[1], destination_image.shape[0]))

    mask = np.zeros_like(destination_image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_points.astype(int), (255, 255, 255))

    mask_inv = cv2.bitwise_not(mask)
    destination_bg = cv2.bitwise_and(destination_image, mask_inv)
    warped_fg = cv2.bitwise_and(warped_image, mask)
    result = cv2.add(destination_bg, warped_fg)
    return result

def add_slant(image, radians = 60):
    height, width = image.shape[:2]
    original_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    tilt_amount = -0.3  
    new_points = np.float32([
        [width * tilt_amount, 0],               
        [width * (1 - tilt_amount), 0],          
        [0, height],                             
        [width, height]                         
    ])
    matrix = cv2.getPerspectiveTransform(original_points, new_points)
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))
    return transformed_image

scene_im = cv2.imread("./field_2.jpeg")
obj_im = cv2.imread("./field_map.jpeg")
obj_im_cp = obj_im.copy()
ad_im = cv2.imread("./logo.png")
ad_im_cp = ad_im.copy()
ad_im = rescale_frame(ad_im, 20)
# ad_im = add_slant(ad_im)

ad_pts = get_advert_coords_from_midpt(ad_im, mp)


scene_marked = mark_pts(scene_im, scene_pts)
obj_marked = mark_pts(obj_im, obj_pts)

h, status = cv2.findHomography(np.array(obj_pts), np.array(scene_pts), cv2.RANSAC)

tx_im = obj_im_cp.copy()
ref_im = scene_marked.copy()

sz = (ref_im.shape[1], ref_im.shape[0])
im_dst = cv2.warpPerspective(tx_im, h, sz)

ad_im_sz = (ad_im_cp.shape[1]*2, ad_im_cp.shape[0]*2)
ad_im_warped = cv2.warpPerspective(ad_im_cp, h, ad_im_sz)

obj_pts_tx = perspective_tx_cv(obj_pts, h)
ad_pts_tx = perspective_tx_cv(ad_pts, h)

warped_marked = mark_pts(im_dst, obj_pts_tx)
warped_marked = mark_pts(warped_marked, ad_pts_tx, marker_color=(255, 0, 255), marker_type=cv2.MARKER_TRIANGLE_DOWN, marker_sz=20, text_sz=1)
warped_adv = add_perspective(ad_im, scene_im, ad_pts_tx)

print(scene_marked.shape)
cv2.imshow("Scene Image", scene_marked)
cv2.imshow("Object -> Scene H-Transform", warped_marked)
cv2.imshow("Markings on scene H transform", warped_adv)
cv2.waitKey(0)
