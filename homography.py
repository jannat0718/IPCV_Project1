# import cv2
# import numpy as np

# obj_pts = [
#     [100, 285],             # GL0
#     [140, 285],             # GL1
#     [140, 415],             # GL2
#     [100, 415],             # GL3
#     [100, 210],             # OB0
#     [100, 50]               # P0
# ]

# scene_pts = [
#     [731, 344],
#     [1029, 367],
#     [361, 537],
#     [28, 507],
#     [1062, 268],
#     [1400, 188]
# ]

# mp = (70, 210)

# def rescale_frame(frame_input, percent=75):    
#     width = int(frame_input.shape[1] * percent / 100)    
#     height = int(frame_input.shape[0] * percent / 100)    
#     dim = (width, height)    
#     return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

# def get_advert_coords_from_midpt(ad_im, midpt):
#     axh = ad_im.shape[0] // 2
#     ayh = ad_im.shape[1] // 2

#     return np.array([
#         [midpt[0] + axh, midpt[1] + ayh],
#         [midpt[0] + axh, midpt[1] - ayh],
#         [midpt[0] - axh, midpt[1] - ayh],
#         [midpt[0] - axh, midpt[1] + ayh],
#     ])

# def mark_pts(im, pts, marker_color = (255, 50, 0), text_color = (0, 50, 255), marker_type = cv2.MARKER_CROSS, text_sz = 2, marker_sz = 30):
#     for i, p in enumerate(pts):
#         cv2.drawMarker(im, tuple(p), marker_color, marker_type, marker_sz, 3)
#         cv2.putText(im, str(i), (p[0], p[1] - 10), cv2.FONT_HERSHEY_COMPLEX, text_sz, text_color, 2)
#     return im

# scene_im = cv2.imread("./field_2.jpeg")
# obj_im = cv2.imread("./field_map.jpeg")
# ad_im = cv2.imread("./UCL_2.png")
# ad_im = rescale_frame(ad_im, 10)



# scene_marked = mark_pts(scene_im, scene_pts)
# obj_marked = mark_pts(obj_im, obj_pts)
# ad_marked = mark_pts(obj_im, get_advert_coords_from_midpt(ad_im, mp), marker_color=(255, 0, 255), marker_type=cv2.MARKER_TRIANGLE_DOWN, marker_sz=15, text_sz=0.5)

# h, status = cv2.findHomography(np.array(obj_pts), np.array(scene_pts), cv2.RANSAC)

# tx_im = obj_im.copy()
# ref_im = scene_marked.copy()

# sz = (ref_im.shape[1], ref_im.shape[0])
# im_dst = cv2.warpPerspective(tx_im, h, sz)





# print(scene_marked.shape)
# cv2.imshow("Scene Image", scene_marked)
# cv2.imshow("Object -> Scene H-Transform", im_dst)
# cv2.imshow("Object Image", ad_marked)
# cv2.waitKey(0)


# import cv2
# import numpy as np

# obj_pts = [
#     [100, 285],             # GL0
#     [140, 285],             # GL1
#     [140, 415],             # GL2
#     [100, 415],             # GL3
#     [100, 210],             # OB0
#     [100, 50]               # P0
# ]

# scene_pts = [
#     [731, 344],
#     [1029, 367],
#     [361, 537],
#     [28, 507],
#     [1062, 268],
#     [1400, 188]
# ]

# mp = (50, 210)

# def rescale_frame(frame_input, percent=75):    
#     width = int(frame_input.shape[1] * percent / 100)    
#     height = int(frame_input.shape[0] * percent / 100)    
#     dim = (width, height)    
#     return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

# def get_advert_coords_from_midpt(ad_im, midpt):
#     axh = ad_im.shape[0] // 2
#     ayh = ad_im.shape[1] // 2

#     return np.array([
#         [midpt[0] + axh, midpt[1] + ayh],
#         [midpt[0] + axh, midpt[1] - ayh],
#         [midpt[0] - axh, midpt[1] - ayh],
#         [midpt[0] - axh, midpt[1] + ayh],
#     ])

# def mark_pts(im, pts, marker_color = (255, 50, 0), text_color = (0, 50, 255), marker_type = cv2.MARKER_CROSS):
#     for i, p in enumerate(pts):
#         cv2.drawMarker(im, tuple(p), marker_color, marker_type, 30, 3)
#         cv2.putText(im, str(i), (p[0], p[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 2, text_color, 3)
#     return im


# def perspective_tx(obj_pts, H):
#     norm_pts = []
#     for p in obj_pts:
#         norm_pts.append([p[0], p[1], 1])
#     tx_pts = []
#     for p in norm_pts:
#         tp = np.array(H).dot(np.array(p).T)
#         tx_pts.append([tp[0], tp[1]])
#     return np.array(tx_pts)


# scene_im = cv2.imread("./field_2.jpeg")
# obj_im = cv2.imread("./field_map.jpeg")
# ad_im = cv2.imread("./UCL_2.png")
# ad_im = rescale_frame(ad_im, 20)



# scene_marked = mark_pts(scene_im, scene_pts)
# obj_marked = mark_pts(obj_im, obj_pts)

# h, status = cv2.findHomography(np.array(obj_pts), np.array(scene_pts), cv2.RANSAC)
# #perspective_transform = cv2.getPerspectiveTransform(np.float32(obj_pts), np.float32(scene_pts))  

# tx_im = obj_im.copy()
# ref_im = scene_marked.copy()

# sz = (ref_im.shape[1], ref_im.shape[0])
# im_dst = cv2.warpPerspective(tx_im, h, sz)


# transformed_points = perspective_tx(obj_pts, h)
# print(transformed_points)
# #transformed_ad_points = cv2.perspectiveTransform(get_advert_coords_from_midpt(ad_im, mp), h)
# warped_marked = mark_pts(im_dst, transformed_points)
# #ad_marked = mark_pts(im_dst, transformed_ad_points, marker_color=(255, 0, 255), marker_type=cv2.MARKER_TRIANGLE_DOWN)




# print(scene_marked.shape)
# cv2.imshow("Scene Image", scene_marked)
# cv2.imshow("Object -> Scene H-Transform", im_dst)
# cv2.imshow("Object Image", warped_marked)
# cv2.waitKey(0)