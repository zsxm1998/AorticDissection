# class Elem():
#     def __init__(self, key, contour):
#         self.root = key
#         self.end = key
#         self.contours = [contour]
        
#     def __len__(self):
#         return len(self.contours)
    
#     def append(self, key, contour):
#         self.end = key
#         self.contours.append(contour)
        
#     def get_bboxes(self):
#         self.circles = [cv2.minEnclosingCircle(c) for c in self.contours]
#         def process(circle):
#             cx, cy, r = round(circle[0][0]), round(circle[0][1]), circle[1]
#             return cx - int(1.5*r), cy - int(1.5*r), cx + int(1.5*r), cy + int(1.5*r)
            
#         self.bboxes = [process(c) for c in self.circles]
        
#     def __str__(self):
#         return str(self.root)+'->'+str(self.end)+':'+str(self.contours)
    
#     def get_prev_area(self):
#         return cv2.contourArea(self.contours[-2])
    
#     @staticmethod
#     def calc_ratio(proportion, kind=0, lower=0.08, upper=0.12):
#         if proportion >= upper:
#             return 1 if kind == 0 else 1 if kind == 1 else 1.2
#         elif proportion <= lower:
#             return 3 if kind == 0 else 8 if kind == 1 else 1.5
#         else:
#             if kind == 0:
#                 return 2*sqrt((upper-proportion)/(upper-lower))+1 
#             elif kind == 1:
#                 return (upper-proportion)/(upper-lower)*7+1
#             else:
#                 return 0.3*sqrt((upper-proportion)/(upper-lower))+1.2
    
#     def post_process(self, patient_pixels, start_x, start_y):       
#         self.post_bboxes = []
#         img_height, img_width = patient_pixels.shape[1:3]
#         start, end = path.root[0], path.end[0]+1
#         for i in range(start, end):
#             #cx, cy, r = round(self.circles[i-start][0][0]+start_x), round(self.circles[i-start][0][1]+start_y), int(self.circles[i-start][1])
#             cx = (self.bboxes[i-start][2]+self.bboxes[i-start][0])//2 + start_x
#             cy = (self.bboxes[i-start][3]+self.bboxes[i-start][1])//2 + start_y
#             r = int((self.bboxes[i-start][2]-self.bboxes[i-start][0])/3)
#             sx, sy = max(0, int(cx-4.5*r)), max(0, int(cy-4.5*r))
#             ex, ey = min(img_width, int(cx+4.5*r+1)), min(img_height, int(cy+4.5*r+1))
#             gray = patient_pixels[i, sy:ey, sx:ex].astype(np.float32)
#             gray = np.clip(gray, 0, 100)
#             gray = (gray-gray.min())/(gray.max()-gray.min())*255
#             gray = gray.astype(np.uint8)
#             gray = cv2.medianBlur(gray, 7)
#             # gray = cv2.GaussianBlur(gray, (7,7), 2)
#             lap = cv2.Laplacian(gray, cv2.CV_16S)
#             gray = gray.astype(np.int16) - lap
#             gray = np.clip(gray, 0, 255).astype(np.uint8)
#             proportion = r*3 / min(img_height, img_width)
#             fc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min(*gray.shape)//2, param1=200, param2=20, minRadius=r, maxRadius=int(r*self.calc_ratio(proportion))+1)
#             if fc is None:
#                 #self.post_bboxes.append((self.bboxes[i-start][0]+start_x, self.bboxes[i-start][1]+start_y, self.bboxes[i-start][2]+start_x, self.bboxes[i-start][3]+start_y))
#                 ratio = self.calc_ratio(proportion, kind=2)
#                 m, n, p, q = cx - int(ratio*r), cy - int(ratio*r), cx + int(ratio*r), cy + int(ratio*r)
#                 self.post_bboxes.append((m, n, p, q))
#             else:
#                 if len(fc) != 1:
#                     raise Exception(f'len(fc) == {len(fc)}')
#                 fc = fc[0]
#                 for circle in fc:
#                     fx, fy, fr = round(circle[0]+sx), round(circle[1]+sy), int(circle[2])
#                     assert fr >= r
#                     dis = sqrt((fx-cx)**2+(fy-cy)**2)
#                     if 0.7*(fr-r) < dis < fr - r/self.calc_ratio(proportion, 1):
#                         m, n, p, q = fx-int(1.3*fr), fy-int(1.3*fr), fx+int(1.3*fr), fy+int(1.3*fr)
#                         self.post_bboxes.append((m, n, p, q))
#                         break
#                 else:
#                     #self.post_bboxes.append((self.bboxes[i-start][0]+start_x, self.bboxes[i-start][1]+start_y, self.bboxes[i-start][2]+start_x, self.bboxes[i-start][3]+start_y))
#                     ratio = self.calc_ratio(proportion, kind=2)
#                     m, n, p, q = cx - int(ratio*r), cy - int(ratio*r), cx + int(ratio*r), cy + int(ratio*r)
#                     self.post_bboxes.append((m, n, p, q))
                    
#     def blur_bboxes(self, kind=0):
#         if kind == 0:
#             bboxes = self.bboxes
#         else:
#             bboxes = self.post_bboxes
            
#         xs = [(x1+x2)//2 for x1, y1, x2, y2 in bboxes]
#         ys = [(y1+y2)//2 for x1, y1, x2, y2 in bboxes]
#         ws = [x2-x1 for x1, y1, x2, y2 in bboxes]
#         ksize, thresh = 10, 0.17
#         if len(bboxes) < 300:
#             ksize, thresh = 5, 0.25
#         elif len(bboxes) < 500:
#             ksize, thresh = 8, 0.20
#         for i in range(ksize, len(bboxes)-ksize):
#             mxs = sorted(xs[i-ksize: i+ksize+1])
#             mys = sorted(ys[i-ksize: i+ksize+1])
#             mws = sorted(ws[i-ksize: i+ksize+1])
#             cx, cy, cw = mxs[ksize], mys[ksize], mws[ksize]
#             if i == 442:
#                 print(abs(cw-ws[i]) / cw)
#             if abs(cw-ws[i]) / cw > thresh:
#                 r = cw//2
#                 bboxes[i] = (cx-r, cy-r, cx+r, cy+r)

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

