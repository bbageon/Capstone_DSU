import math

def cal_rad(arr1, arr2, img_center_x, img_center_y):
    # 이미지 중심에서 각 점까지의 벡터 계산
    vector1 = (arr1[0] - img_center_x, arr1[1] - img_center_y)
    vector2 = (arr2[0] - img_center_x, arr2[1] - img_center_y)

    # 각 벡터의 크기 계산
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # 두 벡터의 내적 계산
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # 코사인 법칙을 이용하여 각도 계산
    cos_angle = dot_product / (magnitude1 * magnitude2)

    # 각도를 라디안에서 도로 변환
    angle = math.acos(cos_angle)
    angle_deg = math.degrees(angle)
    
    # 벡터의 외적 계산
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # 각도를 y축을 기준으로 왼쪽은 음수, 오른쪽은 양수값으로 변환
    if cross_product > 0:
        angle_deg = -angle_deg
    
    return angle_deg

# 예시
img_center_x = 0
img_center_y = 0
arr1 = (1, 1)
arr2 = (-1, 1)
print(cal_rad(arr1, arr2, img_center_x, img_center_y))  # Output: -90.0