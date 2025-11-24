# 다양한 필터 적용하기
# 이미지 회전, 확대, 축소 마스터하기
# Affine 변환으로 이미지 왜곡하기
# Perspective 변환으로 원근감 표현하기
# 실전 문제 해결하기

# 필요한 라이브러리 불러오기
import cv2  # OpenCV 라이브러리 - 컴퓨터 비전 작업용
import numpy as np  # 넘파이 - 배열과 행렬 계산용
import matplotlib.pyplot as plt  # 맷플롯립 - 이미지 시각화용
import urllib.request  # URL에서 파일 다운로드용
plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 함수 정의
def create_sample_image(): # 400x400 픽셀 샘플 이미지 생성
    # 흰색 배경 이미지 생성 (400x400 크기, 3채널 컬러)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # 255 = 흰색

    # 파란색 사각형 그리기 (왼쪽 위)
    cv2.rectangle(
        img,  # 그릴 이미지
        (50, 50),  # 시작점 (x, y)
        (150, 150),  # 끝점 (x, y)
        (255, 0, 0),  # BGR 색상 (파란색)
        -1  # -1 = 내부 채우기, 양수 = 테두리 두께
    )

    # 초록색 원 그리기 (오른쪽 위)
    cv2.circle(
        img,  # 그릴 이미지
        (300, 100),  # 중심점 (x, y)
        50,  # 반지름
        (0, 255, 0),  # BGR 색상 (초록색)
        -1  # 내부 채우기
    )

    # 빨간색 삼각형 그리기 (아래쪽)
    triangle_pts = np.array([[200, 250], [150, 350], [250, 350]], dtype=np.int32)  # 세 꼭지점
    cv2.fillPoly(
        img,  # 그릴 이미지
        [triangle_pts],  # 다각형 점들 (리스트로 감싸야 함)
        (0, 0, 255)  # BGR 색상 (빨간색)
    )

    # 텍스트 추가
    cv2.putText(
        img,  # 그릴 이미지
        'OpenCV 2025',  # 표시할 텍스트
        (100, 250),  # 텍스트 시작 위치 (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # 폰트 종류
        1,  # 폰트 크기 (배율)
        (0, 0, 0),  # BGR 색상 (검은색)
        2,  # 텍스트 두께
        cv2.LINE_AA  # 안티앨리어싱 (부드러운 선)
    )

    return img  # 생성된 이미지 반환

# 필터 실습
sample_img = create_sample_image() # 이미지 생성
sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB) # BGR을 RGB 변환
# 1) 평균 블러(Average Blur) : 커널 크기만큼의 픽셀들의  평균값으로 대체
blur_avg = cv2.blur(sample_img,(11,11))  # 커널 크기(가로, 세로), 클수록 더 흐림
# 2) 가우시안 필터(Gaussian Blur) : 평균, 중심에 가까울 수록 더 큰 가중치(가중평균) 주는 블러
blur_gaussian = cv2.GaussianBlur(sample_img,(11,11),0)  # 반드시 홀수여야 함 # 표준편차(0이면 자동계산)
# 3) 중간값 블러(Median Blur) : 평균의 단점(noise)에 효과적
blur_median = cv2.medianBlur(sample_img,11)  # 커널 크기(홀수)
# 4) 양방향 필터(bilateral Filter)
blur_bilateral = cv2.bilateralFilter(sample_img, 15, 75, 75)  # 픽셀 이웃 직경, # 색상 공간의 표준편차, # 좌표 공간의 표준편차

# 4개의 결과를 2x2 그리드로 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2행 2열 서브플롯 생성
fig.suptitle('Blur Filters Comparison', fontsize=16, fontweight='bold')  # 전체 제목

# 각 필터 결과를 RGB로 변환하여 표시
images = [blur_avg, blur_gaussian, blur_median, blur_bilateral]  # 이미지 리스트
titles = ['Average Blur', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter']  # 제목 리스트

for idx, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
    # axes.flat : 이미지가 그려질 위치(grid 에서 )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR >> RGB
    ax.imshow(img_rgb)  # 이미지 표시
    ax.set_title(title, fontsize=12)  # 제목 설정
    ax.axis('off')  # 축 숨기기

plt.tight_layout()  # 서브플롯 간 간격 조정
plt.show()  # 화면에 표시

# 엣지 검출 필터

#1)소벨(sobel) 필터 : 가로/세로 방향 엣지 검출
sobel_x = cv2.Sobel(sample_img,cv2.CV_64F,1,0,ksize=3)
# 출력 이미지 타입 지정(64비트 float)
# x방향 미분 차수(1 = 1차 미분)
# y방향 미분 차수(0 = 미분 안함)
# 커널 사이즈(1,3,5,7,... 홀수에서 하나 선택)
sobel_x = np.uint8(np.absolute(sobel_x))
sobel_y = cv2.Sobel(sample_img,cv2.CV_64F,0,1,ksize=3)
sobel_y = np.uint8(np.absolute(sobel_y))
sobel_combined = cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
# 1번째 이미지와 가중치, 2번째 이미지와 가중치, 감마값(밝기 조절)

# 2) 라플라시안 필터 (Laplacian Filter) : 2차 미분 (곡률, 변환점) >> 모든 방향 엣지 검출
laplacian = cv2.Laplacian(sample_img,cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))    

# 3) 캐니 엣지 (Canny) : 가장 정밀한 엣지 검출, 흑백 전환 필수
gray_img = cv2.cvtColor(sample_img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray_img,100,200) #100_200 사이 값이 200이상 엣지와 연결되어있지 않은 경우 엣지로 인식x

# 4개의 결과를 2x2 그리드로 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2행 2열 서브플롯
fig.suptitle('Edge Detection Filters', fontsize=16, fontweight='bold')  # 전체 제목

# 결과 표시
edge_images = [sobel_combined, laplacian, canny]  # 이미지 리스트
edge_titles = ['Sobel (X+Y)', 'Laplacian', 'Canny Edge', 'Original (Gray)']  # 제목 리스트

for idx, (ax, img, title) in enumerate(zip(axes.flat, edge_images, edge_titles)):
    # 캐니와 그레이는 이미 흑백이므로 cmap='gray' 사용
    if idx >= 2:  # 캐니와 원본 그레이
        ax.imshow(img, cmap='gray')  # 흑백으로 표시
    else:  # 소벨과 라플라시안
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        ax.imshow(img_rgb)  # 컬러로 표시

    ax.set_title(title, fontsize=12)  # 제목 설정
    ax.axis('off')  # 축 숨기기

# 회전, 크기 조절
height, width = sample_img.shape[:2]
center = (width // 2, height//2) # 이미지 중심점 계산 (// = 정수 나눗셈)
# 1) 45도 회전(크기 유지)
matrix_45 = cv2.getRotationMatrix2D(center,45,1.0) # 중심점, 각도(반시계방향), 스케일(1.0=원본크기)
rotated_45 = cv2.warpAffine(sample_img, matrix_45, (width, height))
# 2) 90도 회전(크기 유지)
matrix_90 = cv2.getRotationMatrix2D(center,90,1.0) # 중심점, 각도(반시계방향), 스케일(1.0=원본크기)
rotated_90 = cv2.warpAffine(sample_img, matrix_90, (width, height))
# 3) 45도 회전 + 0.5배 축소
matrix_45_half = cv2.getRotationMatrix2D(center,45,0.5) # 중심점, 각도(반시계방향), 스케일(0.5=절반크기)
rotated_45_half = cv2.warpAffine(sample_img, matrix_45_half, (width, height))
# 4) 30도 회전 + 1.5배 확대
matrix_30_1_5 = cv2.getRotationMatrix2D(center,30,1.5) # 중심점, 각도(반시계방향), 스케일(1.5=1.5배크기)
rotated_30_large = cv2.warpAffine(sample_img, matrix_30_1_5, (width, height))

# 결과를 2x2 그리드로 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2행 2열 서브플롯
fig.suptitle('Rotation & Scaling', fontsize=16, fontweight='bold')  # 전체 제목

# 회전 결과들
rotation_images = [rotated_45, rotated_90, rotated_45_half, rotated_30_large]  # 이미지 리스트
rotation_titles = [
    'Rotate 45° (scale=1.0)',  # 45도 회전
    'Rotate 90° (scale=1.0)',  # 90도 회전
    'Rotate 45° (scale=0.5)',  # 45도 회전 + 축소
    'Rotate 30° (scale=1.5)'   # 30도 회전 + 확대
]

# 각 subplot에 이미지 표시
for ax, img, title in zip(axes.flat, rotation_images, rotation_titles):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    ax.imshow(img_rgb)  # 이미지 표시
    ax.set_title(title, fontsize=11)  # 제목 설정
    ax.axis('off')  # 축 숨기기

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 화면에 표시

# 보간법
# 원본 이미지 2배 확대
# 1) INTER_NEAREST : 최근접 이웃 셀 값을 복사 (가장 빠르지만 품질 낮음, 계단 현상)
resized_nearest = cv2.resize(sample_img, dsize=(width*2, height*2), interpolation=cv2.INTER_NEAREST)
# 2) INTER_LINEAR : 주변 2x2 픽셀의 가중 평균 (기본값, 속도와 품질의 균형)
resized_linear = cv2.resize(sample_img, dsize=(width*2, height*2), interpolation=cv2.INTER_LINEAR)
# 3) INTER_CUBIC : 주변 4x4 픽셀의 가중 평균 (품질 우수, 속도 느림)
resized_cubic = cv2.resize(sample_img, dsize=(width*2, height*2), interpolation=cv2.INTER_CUBIC)
# 4) INTER_LANCZOS4 : 주변 8x8 픽셀의 가중 평균 (최고 품질, 가장 느림)
resized_lanczos = cv2.resize(sample_img, dsize=(width*2, height*2), interpolation=cv2.INTER_LANCZOS4)

# 각 방법의 일부분을 확대해서 비교 (차이를 명확히 보기 위해)
# 중앙 부근의 100x100 픽셀 영역 추출
crop_y, crop_x = 350, 350  # 자를 위치 (확대된 이미지 기준)
crop_size = 100  # 자를 크기

cropped_nearest = resized_nearest[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
cropped_linear = resized_linear[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
cropped_cubic = resized_cubic[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
cropped_lanczos = resized_lanczos[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

# 결과를 2x2 그리드로 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2행 2열 서브플롯
fig.suptitle('Interpolation Methods Comparison (2x Zoom, Cropped)',
             fontsize=16, fontweight='bold')  # 전체 제목

# 확대 결과들 (일부 영역만)
interp_images = [cropped_nearest, cropped_linear, cropped_cubic, cropped_lanczos]
interp_titles = [
    'NEAREST (fastest, lowest quality)',  # 가장 빠름
    'LINEAR (default, balanced)',  # 기본값
    'CUBIC (slow, high quality)',  # 고품질
    'LANCZOS4 (slowest, best quality)'  # 최고품질
]

# 각 subplot에 이미지 표시
for ax, img, title in zip(axes.flat, interp_images, interp_titles):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    ax.imshow(img_rgb)  # 이미지 표시
    ax.set_title(title, fontsize=10)  # 제목 설정
    ax.axis('off')  # 축 숨기기

plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 화면에 표시

# 아핀 변환 : 3개 점을 이용한 변환(삼각형 모서리)

src_pts = np.array([[50,50], [300,50], [50,300]], dtype=np.float32) # 원본 3점
dst_pts = np.array([[10,100], [300,50], [100,250]], dtype=np.float32) # 목표 3점
M = cv2.getAffineTransform(src_pts, dst_pts)
affine_result = cv2.warpAffine(sample_img, M, (width, height)) # 아핀 변환 적용

# 원본 이미지에 점 표시
original_with_pts = sample_img.copy()
for pt in src_pts:
    cv2.circle(original_with_pts, tuple(map(int,pt)), 5, (0,0,0), -1) # 검은색 점 그리기(BGR)
    # 점 위치 (반복되는 pt(point)를 정수로 변환 >> tuple로 데이터 타입 변경)

# 변환된 이미지에 점 표시
affine_with_pts = affine_result.copy()
for pt in dst_pts:
    cv2.circle(affine_with_pts, tuple(map(int,pt)), 5, (0,0,255), -1) # 빨간색 점 그리기(BGR)
    # 점 위치 (반복되는 pt(point)를 정수로 변환 >> tuple로 데이터 타입 변경)   

# 결과를 나란히 표시
fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # 1행 2열 서브플롯
fig.suptitle('Affine Transformation (3 Points)', fontsize=16, fontweight='bold')

# 원본 (파란 점)
axes[0].imshow(cv2.cvtColor(original_with_pts, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (Blue Points)', fontsize=12)
axes[0].axis('off')

# 변환 결과 (빨간 점)
axes[1].imshow(cv2.cvtColor(affine_with_pts, cv2.COLOR_BGR2RGB))
axes[1].set_title('Affine Transformed (Red Points)', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show() 

# 수동으로 Affine 변환 행렬 만들기
# 행렬 구조: [[a, b, tx], [c, d, ty]]
# a, d: 크기 조절
# b, c: 기울이기 (shear)
# tx, ty: 이동

# 1) 단순 이동 (Translation)
translate_matrix = np.float32([
    [1, 0, 50],   # x축: 크기 유지, 기울임 없음, 오른쪽 50픽셀 이동
    [0, 1, 30]    # y축: 크기 유지, 기울임 없음, 아래 30픽셀 이동
])
translated = cv2.warpAffine(sample_img, translate_matrix, (width, height))

# 2) 수평 기울이기 (Horizontal Shear)
shear_x_matrix = np.float32([
    [1, 0.3, 0],  # x축: 크기 유지, y값에 따라 x 이동 (기울임), 이동 없음
    [0, 1, 0]     # y축: 변화 없음
])
sheared_x = cv2.warpAffine(sample_img, shear_x_matrix, (width + 150, height))
# 가로 크기를 늘려서 잘리지 않게 함

# 3) 수직 기울이기 (Vertical Shear)
shear_y_matrix = np.float32([
    [1, 0, 0],    # x축: 변화 없음
    [0.3, 1, 0]   # y축: 크기 유지, x값에 따라 y 이동 (기울임), 이동 없음
])
sheared_y = cv2.warpAffine(sample_img, shear_y_matrix, (width, height + 150))
# 세로 크기를 늘려서 잘리지 않게 함

# 4) 복합 변환 (회전 + 크기 + 이동)
# cos(30°) ≈ 0.866, sin(30°) ≈ 0.5
angle_rad = np.radians(30)  # 30도를 라디안으로 변환
cos_val = np.cos(angle_rad)  # 코사인 값 계산
sin_val = np.sin(angle_rad)  # 사인 값 계산
scale = 0.8  # 0.8배 축소
complex_matrix = np.float32([
    [cos_val * scale, -sin_val * scale, 50],  # 회전 + 축소 + 이동
    [sin_val * scale, cos_val * scale, 80]    # 회전 + 축소 + 이동
])
complex_transformed = cv2.warpAffine(sample_img, complex_matrix, (width, height))

# 변환 결과들
affine_results = [translated, sheared_x, sheared_y, complex_transformed]
affine_titles = [
    'Translation (50, 30)',  # 이동
    'Horizontal Shear (0.3)',  # 수평 기울임
    'Vertical Shear (0.3)',  # 수직 기울임
    'Rotate 30° + Scale 0.8 + Translate'  # 복합 변환
]

# 각 subplot에 이미지 표시
for ax, img, title in zip(axes.flat, affine_results, affine_titles):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Perspective 변환 - 4개의 점을 이용한 원근 변환 : 비스듬하게 찍힌 사진을 정면으로 보정할 때 사용
# 원본 이미지의 4개 꼭지점 (사각형)
src_pts_persp = np.float32([
    [0, 0],           # 왼쪽 위
    [width-1, 0],     # 오른쪽 위
    [width-1, height-1],  # 오른쪽 아래
    [0, height-1]     # 왼쪽 아래
])# 목표 위치 4개 점 (사다리꼴 모양으로 변환)
dst_pts_persp = np.float32([
    [50, 100],        # 왼쪽 위 → 오른쪽+아래 이동
    [width-50, 100],  # 오른쪽 위 → 왼쪽+아래 이동
    [width-20, height-50],  # 오른쪽 아래 → 왼쪽+위 이동
    [20, height-50]   # 왼쪽 아래 → 오른쪽+위 이동
])
# 결과: 위쪽이 좁고 아래쪽이 넓은 사다리꼴 (원근감)

# Perspective 변환 행렬 계산 (3x3 행렬)
perspective_matrix = cv2.getPerspectiveTransform(src_pts_persp,dst_pts_persp)
perspective_result = cv2.warpPerspective(sample_img,perspective_matrix,(width, height),borderValue=(200, 200, 200) ) # 빈 공간을 회색으로 채움

# 시각화: 원본과 목표 점들을 선으로 연결
original_persp = sample_img.copy()
perspective_persp = perspective_result.copy()

# 원본 이미지에 점과 선 그리기
for i, pt in enumerate(src_pts_persp):
    # 점 그리기
    cv2.circle(original_persp,tuple(pt.astype(int)),8,(255, 0, 0),-1,)
    # 점 번호 표시
    cv2.putText(original_persp,str(i+1),tuple((pt + [10, 10]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2)
# 사각형 테두리 그리기
cv2.polylines(original_persp,[src_pts_persp.astype(int)],True,(255, 0, 0), 3)

# 변환된 이미지에 점과 선 그리기
for i, pt in enumerate(dst_pts_persp):
    cv2.circle(
        perspective_persp, # 이미지
        tuple(pt.astype(int)), # 점 위치
        8, # 직경
        (0, 0, 255),  # 빨간색
        -1 # 내부 채우기
    )
    cv2.putText(
        perspective_persp, # 이미지
        str(i+1), # 텍스트
        tuple((pt + [10, 10]).astype(int)), # 텍스트 위치
        cv2.FONT_HERSHEY_SIMPLEX, # 폰트
        0.7, # 폰트 크기
        (0, 0, 255),
        2 # 두께
    )
cv2.polylines(
    perspective_persp, # 이미지
    [dst_pts_persp.astype(int)], # 점들을 int로 변환
    True, # 닫힌 도형
    (0, 0, 255),  # 빨간색
    3 # 두께
)

# 결과 표시
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Perspective Transformation (4 Points)', fontsize=16, fontweight='bold')

# 원본 (파란색 사각형)
axes[0].imshow(cv2.cvtColor(original_persp, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (Blue Rectangle)', fontsize=12)
axes[0].axis('off')

# 변환 결과 (빨간색 사다리꼴)
axes[1].imshow(cv2.cvtColor(perspective_persp, cv2.COLOR_BGR2RGB))
axes[1].set_title('Perspective Transformed (Red Trapezoid)', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 실전 예제 : 원본을 먼저 비스듬하게 만들고, 다시 정면으로 복원

# Step 1: 정면 → 비스듬하게 (원근 효과 추가)
src_straight = np.float32([[50, 50],[width-50, 50],[width-50, height-50],[50, height-50]])
dst_skewed = np.float32([
    [100, 50],       # 왼쪽 위
    [width-50, 80],  # 오른쪽 위
    [width-80, height-100],  # 오른쪽 아래
    [80, height-80]  # 왼쪽 아래
])
M_skew = cv2.getPerspectiveTransform(src_straight, dst_skewed) # 변환행렬
skewed_document = cv2.warpPerspective(sample_img,M_skew,(width, height),borderValue=(255, 255, 255)) # 흰색 배경

# Step 2: 비스듬함 → 정면으로 (보정)
M_correct = cv2.getPerspectiveTransform(dst_skewed, src_straight) # 역 변환행렬
corrected_document = cv2.warpPerspective(skewed_document,M_correct,(width, height),borderValue=(255, 255, 255))

# 3단계 비교: 원본 → 비스듬함 → 보정
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Document Correction Process', fontsize=16, fontweight='bold')
axes[0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('1. Original', fontsize=12)
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(skewed_document, cv2.COLOR_BGR2RGB))
axes[1].set_title('2. Skewed (Camera View)', fontsize=12)
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(corrected_document, cv2.COLOR_BGR2RGB))
axes[2].set_title('3. Corrected (Straightened)', fontsize=12)
axes[2].axis('off')
plt.tight_layout()
plt.show()