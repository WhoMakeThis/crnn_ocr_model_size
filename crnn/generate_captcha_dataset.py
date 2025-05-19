import os
import random
import string
from captcha.image import ImageCaptcha

# ====== 설정 ======
SAVE_DIR = 'dataset'
NUM_IMAGES = 50000
LABEL_LEN = 5
CHARS = string.ascii_uppercase + string.digits

# ====== 폰트 경로 (Windows용) ======
# 역슬래시(\\) 문제 피하기 위해 raw string 사용 (또는 슬래시(/)로 변경)
FONT_PATH = "DejaVuSans-Bold.ttf"

# ====== 폰트 존재 여부 확인 ======
if not os.path.isfile(FONT_PATH):
    raise FileNotFoundError(f"❌ 폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
else:
    print(f"✅ 폰트 확인됨: {FONT_PATH}")

# ====== 이미지 생성기 초기화 ======
image = ImageCaptcha(width=150, height=50, fonts=[FONT_PATH])
os.makedirs(SAVE_DIR, exist_ok=True)
used_labels = set()

# ====== 이미지 생성 루프 ======
for i in range(NUM_IMAGES):
    while True:
        label = ''.join(random.choices(CHARS, k=LABEL_LEN))
        if label not in used_labels:
            used_labels.add(label)
            break
    img = image.generate_image(label)
    img.save(os.path.join(SAVE_DIR, f'{label}.png'))

    if (i+1) % 1000 == 0:
        print(f"{i+1} / {NUM_IMAGES} 생성 완료")

print(f"✅ {NUM_IMAGES}개의 CAPTCHA 이미지가 '{SAVE_DIR}' 폴더에 생성되었습니다.")

