import cv2

# ---- ここがポイント：ファイルパスは必ず "data/xxx.jpg" ----
img = cv2.imread("data/aerial_tsujiwara_1km_z17.jpg")

if img is None:
    print("画像を読み込めませんでした。パスを確認してください。")
    exit(1)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked: x={x}, y={y}")

cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
